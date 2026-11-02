"""Offline /analyze log-boundary probes; no claim about downstream services."""
import json
import logging

import pytest

import app
from tests.test_analyze_source_correction_route import (
    offline, pipeline, synthetic_disclosure_bypass,
)
from tests.test_correction_integration_offline import DRAFT, SOURCE, Response


SENTINEL = "SYNTHETIC_PRIVATE_29F0"


def assert_private_logs_absent(caplog, capsys):
    output = capsys.readouterr()
    assert SENTINEL not in output.out + output.err + caplog.text
    for record in caplog.records:
        assert SENTINEL not in repr(record.__dict__)
        assert not record.exc_info and not record.stack_info


@pytest.mark.parametrize("sse", [False, True])
@pytest.mark.parametrize("outcome", ["approved", "pipeline_error", "correction_error"])
def test_analyze_logs_exclude_request_provider_and_error_content(
    offline, monkeypatch, synthetic_disclosure_bypass, caplog, capsys, sse, outcome,
):
    client, session = offline
    caplog.set_level(logging.DEBUG)
    source = SOURCE.replace("repair procedures", "repair procedures " + SENTINEL)
    draft = DRAFT.replace("repair procedures", "repair procedures " + SENTINEL)
    pipeline(monkeypatch, draft=draft, extra={"private_provider_field": SENTINEL})
    if outcome == "pipeline_error":
        async def fail_pipeline(**kwargs):
            raise RuntimeError(SENTINEL)
        monkeypatch.setattr(app, "run_new_pipeline_async", fail_pipeline)
    elif outcome == "correction_error":
        async def fail_correction(*args, **kwargs):
            raise RuntimeError(SENTINEL)
        monkeypatch.setattr(app, "_run_source_correction", fail_correction)

    headers = {"X-Cogito-Owner-Id": "offline-owner", "X-Request-Id": SENTINEL}
    if sse:
        headers["Accept"] = "text/event-stream"
    response = client.post("/analyze", headers=headers, json={
        "resume_text": source, "job_ad": SENTINEL,
        "extracted_skills_json": json.dumps([{ "skill": SENTINEL }]),
        "domain_insights_json": json.dumps({SENTINEL: {"value": SENTINEL}}),
        "job_title": SENTINEL, "location": SENTINEL,
    })
    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == SENTINEL
    if outcome == "approved":
        assert '"correction_status":"approved"' in response.text.replace(" ", "")
        assert session.requests
    else:
        assert '"correction_status":"withheld"' in response.text.replace(" ", "")
        assert not session.requests
    records = [r for r in caplog.records if r.name == "imaginator"]
    assert any(r.getMessage() == "analyze.request.incoming" for r in records)
    ids = [r.request_id for r in records if getattr(r, "request_id", None)]
    assert ids and len(set(ids)) == 1 and ids[0] != SENTINEL
    assert_private_logs_absent(caplog, capsys)


@pytest.mark.parametrize("sse", [False, True])
@pytest.mark.parametrize("failure", ["body", "transport"])
def test_provider_failure_logs_do_not_include_raw_body_or_exception(
    offline, monkeypatch, synthetic_disclosure_bypass, caplog, capsys, sse, failure,
):
    client, _ = offline
    caplog.set_level(logging.DEBUG)
    pipeline(monkeypatch)

    class FailingSession:
        calls = 0

        async def close(self):
            pass

        def post(self, *args, **kwargs):
            self.calls += 1
            if failure == "transport":
                raise RuntimeError(SENTINEL)
            return Response(('{"' + SENTINEL).encode())

    session = FailingSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    headers = {"Accept": "text/event-stream"} if sse else {}
    response = client.post("/analyze", headers=headers,
                           json={"resume_text": SOURCE, "job_ad": "job"})
    assert response.status_code == 200 and session.calls == 1
    assert '"correction_status":"withheld"' in response.text.replace(" ", "")
    assert_private_logs_absent(caplog, capsys)


@pytest.mark.parametrize("field", ["extracted_skills_json", "domain_insights_json"])
def test_internal_parse_exception_details_not_logged(offline, monkeypatch, caplog, capsys, field):
    client, _ = offline
    caplog.set_level(logging.DEBUG)
    pipeline(monkeypatch, status="failed")
    value = json.dumps({"test_parse": SENTINEL})
    real_loads = json.loads
    calls = 0

    def fail_after_validation(data, *args, **kwargs):
        nonlocal calls
        if data == value:
            calls += 1
            if calls == 2:
                raise ValueError(SENTINEL)
        return real_loads(data, *args, **kwargs)

    monkeypatch.setattr(app.json, "loads", fail_after_validation)
    response = client.post("/analyze", json={"resume_text": SOURCE, "job_ad": "job", field: value})
    assert response.status_code == 200 and calls == 2
    assert response.json()["correction_status"] == "withheld"
    assert any("Could not parse " + field in r.getMessage() for r in caplog.records)
    assert_private_logs_absent(caplog, capsys)


@pytest.mark.parametrize("field", ["extracted_skills_json", "domain_insights_json"])
def test_invalid_json_validation_does_not_log_input(offline, caplog, capsys, field):
    client, session = offline
    caplog.set_level(logging.DEBUG)
    response = client.post("/analyze", json={
        "resume_text": SOURCE, "job_ad": "job", field: '{"' + SENTINEL,
    })
    assert response.status_code == 422 and not session.requests
    assert_private_logs_absent(caplog, capsys)