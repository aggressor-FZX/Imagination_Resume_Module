"""Synthetic, mocked offline tests; no semantic-model or external-service validation."""
import asyncio
import json
import socket

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
import tier_a_verifier
from correction_provider import CorrectionProvider, snapshot_sources
from pipeline_config import OR_SLUG_DRAFTER, estimate_cost
from source_correction import SourceDocument, correct


SOURCE = """## Professional Experience
**Synthetic Workshop**
- Maintained the workshop inventory and documented repair procedures for service technicians, coordinated inspections with the team, and kept complete records for all routine equipment maintenance.
- Improved diagnostic accuracy by 30% using structured test procedures and careful review of repair records, with written documentation shared with colleagues during regular team meetings.

## Education
**Synthetic College**
- Completed technical training in electrical systems, equipment maintenance, safety practices, and troubleshooting with practical instruction and supervised workshops.
"""
DRAFT = SOURCE.replace("30% using", "30% in 6 months using")
FRAGMENT = " in 6 months"
SECOND_FRAGMENT = " over 2 years"


def inventory(payload):
    """Partition the synthetic source into factual lines and their section context."""
    assert set(payload) == {"contract", "sources"}  # No draft in inventory.
    items = []
    for source in payload["sources"]:
        text, sid = source["text"], source["source_id"]
        start = position = 0
        for part in text.splitlines(keepends=True):
            position += len(part)
            if not part.strip() or part.startswith("## "):
                continue
            # Include trailing blank lines in the next factual span, not alone.
            items.append({"id": f"{sid}:{start}:{position}", "source_id": sid,
                          "start": start, "end": position, "quote": text[start:position],
                          "classification": "required"})
            start = position
        assert start == len(text)
    return {"source_claims": items}


def review(payload):
    """Explicit fixture decisions, not an inference about arbitrary text."""
    claims = []
    represented = {item["id"]: [] for item in payload["source_claims"]}
    for line in payload["lines"]:
        text = line["text"]
        if text.startswith("## "):
            claims.append({"id": f"{line['line_id']}:0:{len(text)}", "line_id": line["line_id"],
                           "start": 0, "end": len(text), "verdict": "structural", "citations": []})
            continue
        fragment = next((part for part in (FRAGMENT, SECOND_FRAGMENT) if part in text), None)
        clean = text.replace(fragment, "", 1) if fragment else text
        item = next((item for item in payload["source_claims"]
                     if item["quote"].strip().splitlines()[-1] == clean), None)
        if item is None:
            spans = [(0, len(text), "unsupported")]
        elif fragment:
            start = text.index(fragment)
            spans = [(0, start, "supported"), (start, start + len(fragment), "unsupported"),
                     (start + len(fragment), len(text), "supported")]
        else:
            spans = [(0, len(text), "supported")]
        for a, b, verdict in spans:
            if a == b:
                continue
            claims.append({"id": f"{line['line_id']}:{a}:{b}", "line_id": line["line_id"],
                           "start": a, "end": b, "verdict": verdict,
                           "citations": ([{key: item[key] for key in ("source_id", "start", "end", "quote")}]
                                         if verdict == "supported" else [])})
            if verdict == "supported":
                represented[item["id"]].append({"line_id": line["line_id"], "start": a, "end": b})
    mappings = [{"source_claim_id": item["id"],
                 "status": "represented" if represented[item["id"]] else "missing",
                 "spans": represented[item["id"]]} for item in payload["source_claims"]]
    return {"claims": claims, "representations": mappings}


def indexed_body_inventory(payload):
    """Classify index segments; headings are ignored, blank runs were merged."""
    required, non_resume, uncertain = [], [], []
    for segment in payload["segments"]:
        text = segment["text"]
        if text.lstrip().startswith("## "):
            non_resume.append(segment["index"])
        elif text.strip():
            required.append(segment["index"])
        else:
            uncertain.append(segment["index"])
    return {"required": required, "non_resume": non_resume, "uncertain": uncertain}


def indexed_body_review(payload):
    """Scripted verdicts by exact text, expressed as the index-list protocol.

    Works from the wire payload (``segments`` + ``lines``), not the engine's
    internal ``source_claims``, because the adapter owns the translation.

    ``candidates`` is only a lookup aid, so every supported line also names the
    segment that establishes it in ``evidence``. That is what an honest reviewer
    must do: retrieval proposes, the reviewer decides.
    """
    segments = payload.get("segments", [])
    lines = payload.get("lines", [])
    verdicts = {"supported": [], "unsupported": [], "uncertain": [], "structural": []}
    evidence = {}

    def norm(value):
        return " ".join(value.split())

    for line in lines:
        text = line["text"]
        if text.startswith("## "):
            verdicts["structural"].append(line["line_id"])
            continue
        fragment = next((part for part in (FRAGMENT, SECOND_FRAGMENT) if part in text), None)
        clean = text.replace(fragment, "", 1) if fragment else text
        # The reviewer searches every segment it was shown, not just the retrieval
        # shortlist: an exact source line is evidence regardless of ranking.
        exact = [segment["index"] for segment in segments if segment["text"].strip() == clean]
        # The wire protocol has no per-span verdicts, so the whole line carries the
        # verdict. A line with an invented fragment is reported unsupported (that
        # is the claim repair must remove).
        verdict = "unsupported" if fragment or not exact else "supported"
        verdicts[verdict].append(line["line_id"])
        if verdict == "supported":
            evidence[line["line_id"]] = exact
    represented, missing = [], []
    for segment in segments:
        quote = norm(segment["text"])
        # Headings and blank separators make no factual claim; the engine reports
        # them as their own classification, so they are not coverage questions.
        if not quote or segment["text"].startswith("## "):
            continue
        # The draft states this fact even when the same line also carries an
        # addition that repair removes, so representation is judged on the line
        # text with the scripted fragment taken back out.
        carried = any(
            segment["index"] in line["candidates"]
            and norm(line["text"].replace(
                next((part for part in (FRAGMENT, SECOND_FRAGMENT) if part in line["text"]),
                     ""), "", 1)) == quote
            for line in lines)
        (represented if carried else missing).append(segment["index"])
    verdicts["evidence"] = evidence
    verdicts["represented"], verdicts["missing"] = represented, missing
    return verdicts


class Stream:
    def __init__(self, data):
        self.data = data

    async def iter_chunked(self, size):
        for i in range(0, len(self.data), size):
            yield self.data[i:i + size]


class Response:
    status = 200

    def __init__(self, body):
        self.content = Stream(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeSession:
    def __init__(self, *, timeout=False, finish="stop", malformed=False, repairs=None,
                 missing_representation=False):
        self.requests = []
        self.timeout = timeout
        self.finish = finish
        self.malformed = malformed
        self.repairs = list(repairs) if repairs is not None else []
        self.missing_representation = missing_representation

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        if self.timeout:
            raise asyncio.TimeoutError("private provider error")
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if "flagged_claim_ids" in payload:
            # The wire protocol carries one verdict per line, so a flagged claim
            # spans the whole line and repair returns the corrected line text.
            # Default repair drops the scripted fragment, making the line an exact
            # source copy. An explicit repairs list keeps the line defective: a
            # nonempty entry substitutes that fragment, and "" removes it, which
            # reproduces the original fragment-level repair script exactly.
            if self.repairs:
                scripted = self.repairs.pop(0)

                def repaired(cid):
                    line_id = cid.partition(":")[0]
                    text = next(line["text"] for line in payload["lines"]
                                if line["line_id"] == line_id)
                    if scripted:
                        return (text.replace(FRAGMENT, scripted)
                                .replace(SECOND_FRAGMENT, scripted))
                    return text.replace(FRAGMENT, "").replace(SECOND_FRAGMENT, "")
            else:
                def repaired(cid):
                    line_id = cid.partition(":")[0]
                    text = next(line["text"] for line in payload["lines"]
                                if line["line_id"] == line_id)
                    return text.replace(FRAGMENT, "").replace(SECOND_FRAGMENT, "")
            body = {"edits": [{"claim_id": cid, "replacement": repaired(cid)}
                              for cid in payload["flagged_claim_ids"]]}
        elif "source_claims" in payload or "lines" in payload:
            # Review request translated to the index-list wire protocol.
            body = indexed_body_review(payload)
            if self.missing_representation and body["represented"]:
                # Report the first represented segment as omitted, whatever its
                # index: headings are skipped, so index 0 is not always factual.
                index = body["represented"].pop(0)
                body["missing"] = [value for value in body["missing"] if value != index]
                body["missing"].append(index)
        else:
            body = indexed_body_inventory(payload)
        content = '{"claims":[],"claims":[]}' if self.malformed else json.dumps(body)
        return Response(json.dumps({"model": OR_SLUG_DRAFTER,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"finish_reason": self.finish, "message": {"content": content}}]}).encode())


@pytest.fixture
def setup(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("external network forbidden")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(app.settings, "openrouter_api_key_1", "fake-key")
    monkeypatch.setattr(app.settings, "openrouter_api_key_2", None)
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "offline-signing-secret-123456789")
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key, lambda: "fake-auth")
    yield
    app.app.dependency_overrides.clear()


@pytest.fixture
def synthetic_disclosure_bypass(setup, monkeypatch):
    """Mock the privacy boundary for fake sources and sessions, not production policy."""
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)


def pipeline(monkeypatch, status="completed", draft=DRAFT):
    async def fake(**kwargs):
        return {"pipeline_status": status, "final_written_section_markdown": draft,
                "final_written_section": "stale generated content", "rewritten_resume": "stale generated content",
                "experiences": [{"snippet": "stale generated content"}],
                "metrics_support": {"verdict": "PASS (tier A only)", "checked": True},
                "run_metrics": {"calls": [], "failures": [], "total_prompt_tokens": 0,
                                "total_completion_tokens": 0, "total_tokens": 0,
                                "estimated_cost_usd": 0.0}}
    monkeypatch.setattr(app, "run_new_pipeline_async", fake)


def post(client, *, source=SOURCE, docs=None, sse=False, owner="owner-one"):
    data = {"resume_text": source, "job_ad": "Synthetic equipment maintenance role"}
    if docs is not None:
        data["source_documents"] = docs
    headers = {"X-Cogito-Owner-Id": owner}
    if sse:
        headers["Accept"] = "text/event-stream"
    return client.post("/analyze", json=data, headers=headers)


def test_isolated_provider_correction_rechecks_and_preserves_usage(synthetic_disclosure_bypass):
    session = FakeSession()
    provider = CorrectionProvider("fake-key", session)
    result = asyncio.run(correct(DRAFT, (SourceDocument("resume", SOURCE),), provider))
    assert result.status == "approved" and result.markdown == SOURCE
    assert "not proof" in result.commentary
    assert tier_a_verifier.verify_metrics(result.markdown, SOURCE)["verdict"] == "PASS (tier A only)"
    assert "30%" in result.markdown and FRAGMENT not in result.markdown
    assert (result.review_calls, result.repair_calls) == (2, 1)
    assert [call["phase"] for call in provider.usage["calls"]] == [
        "source_inventory", "initial_review", "repair", "post_patch_review"]
    assert provider.calls == len(session.requests) == 4
    assert provider.usage["total_prompt_tokens"] == 40
    assert provider.usage["total_completion_tokens"] == 16
    assert provider.usage["total_tokens"] == 56
    assert provider.usage["estimated_cost_usd"] == pytest.approx(4 * estimate_cost(OR_SLUG_DRAFTER, 10, 4))
    inventory_payload = json.loads(session.requests[0]["json"]["messages"][1]["content"])
    # The inventory question carries source segments and no draft at all.
    assert set(inventory_payload) == {"contract", "segments"}
    assert not any("lines" in json.loads(req["json"]["messages"][1]["content"])
                   for req in session.requests[:1])
    assert all(req["allow_redirects"] is False and req["json"]["provider"] == {"allow_fallbacks": False}
               and req["json"]["model"] == OR_SLUG_DRAFTER for req in session.requests)


def test_mocked_missing_source_representation_withholds_before_repair(synthetic_disclosure_bypass):
    session = FakeSession(missing_representation=True)
    provider = CorrectionProvider("fake-key", session)
    result = asyncio.run(correct(DRAFT, (SourceDocument("resume", SOURCE),), provider))
    assert (result.status, result.reason, result.markdown) == ("withheld", "source_facts_missing", None)
    assert (result.review_calls, result.repair_calls) == (1, 0)
    assert provider.calls == len(session.requests) == 2
    assert provider.usage["total_tokens"] == 28


@pytest.mark.parametrize("status", ["failed", "completed_with_fallback", "unknown"])
def test_real_route_blocks_fallback_without_call(setup, monkeypatch, status):
    pipeline(monkeypatch, status=status)
    session = FakeSession()
    with TestClient(app.app) as client:
        app.app.state.http_session = session
        result = post(client).json()
    assert "fallback_output" in result["review_reasons"] and result["pipeline_status"] == status
    assert not session.requests and result["publication_receipt"] is None
    assert result["final_written_section_markdown"] is None


@pytest.mark.parametrize("mode", ["timeout", "finish", "malformed"])
def test_provider_failure_withholds_without_retry(synthetic_disclosure_bypass, mode):
    session = FakeSession(timeout=mode == "timeout", finish="length" if mode == "finish" else "stop",
                          malformed=mode == "malformed")
    provider = CorrectionProvider("fake-key", session)
    result = asyncio.run(correct(DRAFT, (SourceDocument("resume", SOURCE),), provider))
    assert result.status == "withheld" and result.markdown is None
    assert "private provider error" not in repr(result)
    assert provider.usage["failures"] and len(session.requests) == 1
    assert provider.calls == 1 and provider.usage["failures"][0]["phase"] == "source_inventory"
    assert provider.usage["total_tokens"] == (0 if mode == "timeout" else 14)
    assert provider.usage["failures"][0]["usage_unknown"] is (mode == "timeout")


def test_exact_source_reconstruction(setup, monkeypatch):
    docs = [{"source_id": "document-0", "text": "part one", "kind": "candidate_document"},
            {"source_id": "document-1", "text": "part two", "kind": "candidate_document"}]
    assert snapshot_sources("part one\n\n---\n\npart two", docs, True)
    assert snapshot_sources("part one\n\npart two", docs, True) is None
    assert snapshot_sources(SOURCE, None, False)[0].text == SOURCE
    assert app._snapshot_original_documents("part one\n\n---\n\npart two", docs, True)
    pipeline(monkeypatch)
    with TestClient(app.app) as client:
        result = post(client, docs=docs).json()
    assert result["review_reason"] == "source_documents_mismatch"
    assert result["publication_receipt"] is None and result["final_written_section_markdown"] is None


def test_mocked_two_patches_and_rereviews_use_six_requests_and_budget(synthetic_disclosure_bypass):
    session = FakeSession(repairs=[SECOND_FRAGMENT, ""])
    provider = CorrectionProvider("fake-key", session)
    async def run():
        result = await correct(DRAFT, (SourceDocument("resume", SOURCE),), provider)
        # The repair path itself must be capped; the extra headroom is reserved for
        # the single bounded completion re-review and is tested separately.
        with pytest.raises(ValueError, match="budget exhausted"):
            await provider("post_patch_review", {"contract": "test"})
        return result
    result = asyncio.run(run())
    assert result.status == "approved" and result.markdown == SOURCE
    assert "30%" in result.markdown
    assert tier_a_verifier.verify_metrics(result.markdown, SOURCE)["verdict"] == "PASS (tier A only)"
    assert (result.review_calls, result.repair_calls) == (3, 2)
    assert [call["phase"] for call in provider.usage["calls"]] == [
        "source_inventory", "initial_review", "repair", "post_patch_review", "repair", "post_patch_review"]
    assert provider.calls == len(session.requests) == 6
    assert provider.usage["total_prompt_tokens"] == 60
    assert provider.usage["total_completion_tokens"] == 24
    assert provider.usage["total_tokens"] == 84
    assert provider.usage["estimated_cost_usd"] == pytest.approx(6 * estimate_cost(OR_SLUG_DRAFTER, 10, 4))
    # The repair path consumes six of the seven permitted calls; exactly one is left
    # for recovery, and it is not reachable once a repair has happened.
    assert provider.calls == correction_provider.MAX_CALLS - 1


def test_deadline_cancels_inflight_transport_without_retry(synthetic_disclosure_bypass):
    class HangingResponse:
        async def __aenter__(self):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def __aexit__(self, *args):
            pass

    class HangingSession:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return HangingResponse()

    async def run():
        nonlocal cancelled
        cancelled = asyncio.Event()
        session = HangingSession()
        provider = CorrectionProvider("fake", session)
        result = await correct(DRAFT, (SourceDocument("resume", SOURCE),), provider,
                               timeout_seconds=0.02)
        await asyncio.wait_for(cancelled.wait(), 1)
        assert result.status == "withheld" and result.reason == "timeout_source_inventory"
        assert session.calls == 1
        assert provider.usage["failures"][0]["usage_unknown"] is True

    cancelled = None
    asyncio.run(run())


@pytest.mark.parametrize("docs", [[], None, [{"source_id": "x", "text": "part one", "kind": "job_ad"}],
    [{"source_id": "x", "text": "part one", "kind": "candidate_document"},
     {"source_id": "x", "text": "part two", "kind": "candidate_document"}]])
def test_invalid_explicit_sources_fail_closed(docs):
    assert snapshot_sources("part one\n\n---\n\npart two", docs, True) is None
    assert app._snapshot_original_documents("part one\n\n---\n\npart two", docs, True) is None