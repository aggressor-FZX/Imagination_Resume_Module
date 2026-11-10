"""Semantic provider boundary; all HTTP traffic uses in-memory doubles."""
import asyncio
import copy
import json
import traceback

import pytest

import correction_provider as cp
from disclosure import redact_text
from semantic_review import json_engine
from source_correction import PrivacyReviewRequired, ProviderCallTimeout, ProviderUnavailable


MODEL = "test/configurable-model"
ANSWER = {"version": cp.SEMANTIC_REVIEW_VERSION, "decision": "approve",
          "rationale": "supported", "issues": []}
PAYLOAD = {"contract": "Evaluate the document.", "text": "Contact a@example.test",
           "sources": [{"source_id": "resume", "text": "source evidence"}]}


class Response:
    def __init__(self, status, body):
        self.status, self.body, self.content, self.headers = status, body, self, {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_chunked(self, size):
        yield self.body.encode()


def envelope(content):
    return json.dumps({"model": MODEL, "usage": {"prompt_tokens": 10,
        "completion_tokens": 5, "total_tokens": 15}, "choices": [
            {"finish_reason": "stop", "message": {"content": content}}]})


class Session:
    def __init__(self, responses):
        self.responses, self.requests, self.timeouts = list(responses), [], []

    def post(self, url, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["json"]))
        self.timeouts.append(kwargs["timeout"].total)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return Response(*item)


def run(provider, payload=PAYLOAD, phase="semantic_review"):
    return asyncio.run(provider(phase, payload))


@pytest.fixture
def approved(monkeypatch):
    monkeypatch.setenv(cp.DISCLOSURE_SETTING, "1")
    async def instant(_):
        pass
    monkeypatch.setattr(cp, "_sleep", instant)


def test_plain_json_redaction_model_and_usage(approved):
    session = Session([(200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    assert run(provider) == ANSWER
    wire = session.requests[0]
    sent = json.loads(wire["messages"][1]["content"])
    assert wire["model"] == MODEL and "response_format" not in wire
    assert sent["text"] == redact_text(PAYLOAD["text"])[0]
    assert "a@example.test" not in json.dumps(wire)
    assert provider.usage["total_tokens"] == 15
    assert provider.calls == provider.attempts == 1


def test_schema_only_on_explicit_opt_in(approved):
    session = Session([(200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL, semantic_schema_supported=True)
    run(provider)
    request = session.requests[0]
    assert request["response_format"]["json_schema"]["schema"] == cp.SEMANTIC_REVIEW_SCHEMA
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["provider"]["require_parameters"] is True


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(draft="a@example.test"),
    lambda p: p.update(_hidden="a@example.test"),
    lambda p: p["sources"][0].update(extra="a@example.test"),
    lambda p: p["sources"][0].update(source_id="a@example.test"),
    lambda p: p["sources"][0].update(source_id="5550100199"),
    lambda p: p.update(text={"text": "a@example.test"}),
])
def test_unknown_or_uncovered_fields_fail_before_post(approved, mutation):
    payload = copy.deepcopy(PAYLOAD)
    mutation(payload)
    session = Session([])
    with pytest.raises(RuntimeError):
        run(cp.CorrectionProvider("fake", session, model=MODEL), payload)
    assert not session.requests


def test_source_policy_still_requires_pre_redacted_sources(approved):
    payload = copy.deepcopy(PAYLOAD)
    payload["sources"][0]["text"] = "a@example.test"
    session = Session([])
    with pytest.raises(PrivacyReviewRequired):
        run(cp.CorrectionProvider("fake", session, model=MODEL), payload)
    assert not session.requests


def test_disclosure_not_implicitly_enabled(monkeypatch):
    monkeypatch.delenv(cp.DISCLOSURE_SETTING, raising=False)
    session = Session([])
    with pytest.raises(PrivacyReviewRequired):
        run(cp.CorrectionProvider("fake", session, model=MODEL))
    assert not session.requests


@pytest.mark.parametrize("content", [
    'Decision: approve', '[]', '{"version":"x","version":"y"}',
    '{"x":NaN}', json.dumps({**ANSWER, "version": "wrong"}),
    json.dumps({**ANSWER, "extra": "bad"}),
    json.dumps({**ANSWER, "issues": [{"passage": "x"}]}),
])
def test_malformed_response_is_not_accepted_and_usage_is_counted(approved, content):
    session = Session([(200, envelope(content))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    with pytest.raises(ValueError):
        run(provider)
    assert provider.attempts == 1 and provider.usage["total_tokens"] == 15


def test_retry_and_caller_format_correction_share_http_ceiling(approved):
    session = Session([(503, ""), (200, envelope("not json")),
                       (200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL, semantic_max_http_attempts=2)
    with pytest.raises(ValueError):
        run(provider)
    with pytest.raises(RuntimeError):
        run(provider)
    assert len(session.requests) == provider.attempts == 2
    assert provider.usage["total_tokens"] == 15
    assert provider.usage["failures"][0]["usage_unknown"] is True


def test_retry_ceiling_applies_inside_one_call(approved):
    session = Session([(503, ""), (503, ""), (200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL, semantic_max_http_attempts=2)
    with pytest.raises(ProviderUnavailable):
        run(provider)
    assert len(session.requests) == provider.attempts == 2


def test_deadline_cannot_be_extended_or_ignored(approved, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(cp.time, "monotonic", lambda: clock[0])
    session = Session([(200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    provider.set_budget(2)
    clock[0] = 103
    provider.set_budget(100)
    with pytest.raises(ProviderCallTimeout):
        run(provider)
    assert not session.requests


def test_attempt_timeout_is_capped_by_remaining_deadline(approved, monkeypatch):
    monkeypatch.setattr(cp.time, "monotonic", lambda: 100.0)
    session = Session([(200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    provider.set_budget(3)
    assert run(provider) == ANSWER
    assert session.timeouts == [3]


def test_expiry_after_success_keeps_usage_but_rejects_result(approved, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(cp.time, "monotonic", lambda: clock[0])

    class SlowSession(Session):
        def post(self, url, **kwargs):
            response = super().post(url, **kwargs)
            clock[0] = 104.0
            return response

    session = SlowSession([(200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    provider.set_budget(3)
    with pytest.raises(ProviderCallTimeout):
        run(provider)
    assert provider.usage["total_tokens"] == 15


def test_contract_is_redacted_in_all_serialized_messages(approved):
    payload = copy.deepcopy(PAYLOAD)
    payload["contract"] = "Contact private@example.test for instructions"
    session = Session([(200, envelope(json.dumps(ANSWER)))])
    run(cp.CorrectionProvider("fake", session, model=MODEL), payload)
    assert "private@example.test" not in json.dumps(session.requests[0])


def test_transport_exception_traceback_does_not_leak(approved):
    secret = "a@example.test"
    session = Session([RuntimeError(secret)])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    with pytest.raises(RuntimeError) as exc:
        run(provider)
    assert secret not in "".join(traceback.format_exception(exc.value))


def test_legacy_strict_json_still_rejects_nonobject(approved):
    session = Session([(200, envelope("[]"))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    payload = {"contract": "test", "sources": [{"source_id": "resume", "text": "source"}]}
    with pytest.raises(ValueError, match="invalid provider response"):
        run(provider, payload, "repair")


def revision(operation):
    issue = {"operation": operation, "problem": "omitted" if operation == "insert_record" else "unsupported",
             "sources": ["rabc"], "why": "reason"}
    if operation == "insert_record":
        issue["destination"] = "sabc"
    else:
        issue["passage"] = "pabc"
    if operation == "replace":
        issue["correction"] = "Managed inventory."
    return {**ANSWER, "decision": "revise", "issues": [issue]}


@pytest.mark.parametrize("operation", ["replace", "delete", "insert_record"])
def test_operation_schema_and_validator_match_engine(approved, operation):
    answer = revision(operation)
    assert json_engine._shape(answer)
    session = Session([(200, envelope(json.dumps(answer)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL, semantic_schema_supported=True)
    assert run(provider) == answer
    # Exercise the actual JSON-serializable request, not only the schema constant.
    wire = json.loads(json.dumps(session.requests[0]))
    schema = wire["response_format"]["json_schema"]["schema"]
    options = schema["properties"]["issues"]["items"]["oneOf"]
    variant = next(x for x in options if x["properties"]["operation"]["enum"] == [operation])
    assert set(variant["required"]) == set(answer["issues"][0])
    assert variant["additionalProperties"] is False
    assert variant["properties"]["sources"]["minItems"] == 1
    assert variant["properties"]["sources"]["maxItems"] == 1
    assert json.dumps(schema, separators=(",", ":")) in wire["messages"][0]["content"]


@pytest.mark.parametrize("operation,mutate", [
    ("replace", lambda i: i.pop("correction")),
    ("replace", lambda i: i.update(correction=None)),
    ("replace", lambda i: i.update(problem="omitted")),
    ("replace", lambda i: i.update(destination="sabc")),
    ("delete", lambda i: i.update(correction="extra")),
    ("delete", lambda i: i.update(problem="unresolved")),
    ("insert_record", lambda i: i.update(passage="pabc")),
    ("insert_record", lambda i: i.update(problem="misattributed")),
    ("insert_record", lambda i: i.pop("destination")),
    ("replace", lambda i: i.update(operation="unknown")),
    ("delete", lambda i: i.update(sources=[])),
    ("delete", lambda i: i.update(sources=["r", "r2"])),
    ("delete", lambda i: i.update(sources=[None])),
    ("delete", lambda i: i.update(passage=None)),
    ("delete", lambda i: i.update(why=None)),
])
def test_invalid_operation_shapes_are_format_errors(approved, operation, mutate):
    answer = revision(operation)
    mutate(answer["issues"][0])
    assert not json_engine._shape(answer)
    provider = cp.CorrectionProvider("fake", Session([(200, envelope(json.dumps(answer)))]), model=MODEL)
    with pytest.raises(cp.SemanticFormatError):
        run(provider)


def test_too_many_issues_is_format_error(approved):
    answer = revision("delete")
    answer["issues"] *= 33
    with pytest.raises(cp.SemanticFormatError):
        run(cp.CorrectionProvider("fake", Session([(200, envelope(json.dumps(answer)))]), model=MODEL))


@pytest.mark.parametrize("location", ["text", "source", "contract"])
@pytest.mark.parametrize("encoded", [
    r'{"document":"private\u0040example.test"}',
    r'{"document":"555\u0030010019"}',
    r'{"private\u0040example.test":"value"}',
    r'{"document":"private\\u0040example.test"}',
    r'private\u0040example.test',
])
def test_encoded_nested_pii_never_reaches_transport(approved, location, encoded):
    payload = copy.deepcopy(PAYLOAD)
    if location == "source":
        payload["sources"][0]["text"] = encoded
    else:
        payload[location] = encoded
    session = Session([])
    with pytest.raises(PrivacyReviewRequired):
        run(cp.CorrectionProvider("fake", session, model=MODEL), payload)
    assert not session.requests


def test_nested_numeric_pii_never_reaches_transport(approved):
    payload = copy.deepcopy(PAYLOAD)
    payload["text"] = '{"document":5550100199}'
    session = Session([])
    with pytest.raises(PrivacyReviewRequired):
        run(cp.CorrectionProvider("fake", session, model=MODEL), payload)
    assert not session.requests


SOURCE = ("## Professional Experience\n**Acme** -- City\n- Managed inventory.\n"
          "## Projects\n## Education\n## Certifications\n## Technical Skills\n")


def engine_run(provider, draft=SOURCE, source=SOURCE):
    return asyncio.run(json_engine.correct(draft, {"private-source": source}, provider))


def test_real_engine_format_retry_then_approval_and_wire_redaction(approved):
    source = SOURCE.replace("Managed inventory.", "Managed inventory. a@example.test")
    session = Session([(200, envelope("not json")), (200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    result = engine_run(provider, source, source)
    assert result.status == "approved" and result.review_calls == 2
    assert provider.attempts == 2
    first, second = [json.loads(json.dumps(req)) for req in session.requests]
    assert "a@example.test" not in json.dumps([first, second])
    one, two = [json.loads(req["messages"][1]["content"]) for req in (first, second)]
    assert json.loads(one["text"])["document"] == redact_text(source)[0]
    assert json.loads(one["sources"][0]["text"])["document"] == redact_text(source)[0]
    assert one["text"] == two["text"] and one["sources"] == two["sources"]
    assert "invalid JSON shape" in two["contract"]


def test_engine_invalid_format_twice_stops(approved):
    session = Session([(200, envelope("bad")), (200, envelope("bad")),
                       (200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    result = engine_run(provider)
    assert result.reason == "malformed_output" and result.review_calls == 2
    assert provider.attempts == len(session.requests) == 2


def test_engine_invalid_operation_then_valid_response(approved):
    bad = revision("replace")
    del bad["issues"][0]["correction"]
    session = Session([(200, envelope(json.dumps(bad))), (200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    result = engine_run(provider)
    assert result.status == "approved" and result.review_calls == 2
    assert len(session.requests) == 2


def test_engine_encoded_nested_document_is_withheld(approved):
    unsafe = SOURCE + r'{"email":"private\u0040example.test"}' + "\n"
    session = Session([])
    result = engine_run(cp.CorrectionProvider("fake", session, model=MODEL), unsafe, unsafe)
    assert result.reason == "privacy_review_required" and result.review_calls == 1
    assert not session.requests


def test_engine_attempt_budget_does_not_trigger_extra_format_retry(approved):
    session = Session([(200, envelope("bad"))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL, semantic_max_http_attempts=1)
    result = engine_run(provider)
    assert result.reason == "provider_error" and result.review_calls == 2
    assert provider.attempts == len(session.requests) == 1


def test_engine_transport_and_format_retries_share_limit(approved):
    session = Session([(503, ""), (200, envelope("bad")),
                       (200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider("fake", session, model=MODEL, semantic_max_http_attempts=2)
    result = engine_run(provider)
    assert result.reason == "provider_error" and result.review_calls == 2
    assert provider.attempts == len(session.requests) == 2
    assert provider.usage["total_tokens"] == 15


def test_engine_invalid_envelope_does_not_trigger_format_retry(approved):
    session = Session([(200, json.dumps({"model": MODEL}))])
    result = engine_run(cp.CorrectionProvider("fake", session, model=MODEL))
    assert result.reason == "provider_error" and result.review_calls == 1
    assert len(session.requests) == 1


def test_engine_size_failure_is_not_format_retry(approved, monkeypatch):
    monkeypatch.setattr(cp, "MAX_REQUEST_BYTES", 100)
    session = Session([])
    result = engine_run(cp.CorrectionProvider("fake", session, model=MODEL))
    assert result.reason == "provider_error" and result.review_calls == 1
    assert not session.requests


def test_engine_response_size_failure_is_not_format_retry(approved, monkeypatch):
    monkeypatch.setattr(cp, "MAX_RESPONSE_BYTES", 100)
    session = Session([(200, envelope(json.dumps(ANSWER)))])
    result = engine_run(cp.CorrectionProvider("fake", session, model=MODEL))
    assert result.reason == "provider_error" and result.review_calls == 1
    assert len(session.requests) == 1


def test_engine_revision_with_real_adapter_and_fresh_review(approved):
    class DynamicSession(Session):
        def post(self, url, **kwargs):
            self.requests.append(copy.deepcopy(kwargs["json"]))
            payload = json.loads(kwargs["json"]["messages"][1]["content"])
            if len(self.requests) == 1:
                draft = json.loads(payload["text"])
                source = json.loads(payload["sources"][0]["text"])
                reply = revision("replace")
                reply["issues"][0].update(passage=draft["records"][0]["passages"][0]["id"],
                                          sources=[source["records"][0]["id"]])
            else:
                reply = ANSWER
            return Response(200, envelope(json.dumps(reply)))
    session = DynamicSession([])
    provider = cp.CorrectionProvider("fake", session, model=MODEL)
    result = engine_run(provider, SOURCE.replace("Managed inventory.", "Managed inventory. invented claim"))
    assert result.status == "approved" and result.markdown == SOURCE
    assert result.repair_calls == 1 and result.review_calls == 2
    assert len(session.requests) == 2