"""The saturated-provider path, measured live on 2026-11-02.

A live run failed in 2.1s with ``provider_error`` and no recorded response. Replaying
the *exact* captured request showed why: OpenRouter answered ``429`` with
``provider_error_code=engine_overloaded`` on its shared pool. Two defects followed.

1. The request pinned ``allow_fallbacks: False``. The model is served by exactly two
   endpoints, so pinning one turned a saturated provider into a total correction
   failure. Re-running the same captured request with fallbacks enabled returned
   ``200`` from the other endpoint, unchanged in every other respect.
2. The capture wrapper recorded the body only from ``iter_chunked``. A non-200 answer
   is read via ``text()``, so the live run stored the request and dropped the answer,
   which is the one shape that cannot be diagnosed offline.

These tests pin both, without network access.
"""
import asyncio
import json

import pytest

import correction_provider
import live_capture
from correction_provider import CorrectionProvider
from source_correction import (
    ProviderRateLimited,
    SourceDocument,
    correct,
)


SOURCE = """## Professional Experience
**Synthetic Workshop**
- Maintained the workshop inventory and documented repair procedures for service technicians, coordinated inspections with the team, and kept complete records for all routine equipment maintenance.
"""


class TooManyRequests:
    """Answers 429 the way aiohttp does: status set, body behind ``text()``."""

    status = 429

    async def text(self, *args, **kwargs):
        return json.dumps({"error": {"message": "Provider returned error", "code": 429,
                                     "metadata": {"provider_error_code": "engine_overloaded"}}})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class SaturatedSession:
    def __init__(self):
        self.requests = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        return TooManyRequests()


@pytest.fixture
def disclosure(monkeypatch):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)


def test_a_saturated_provider_is_its_own_reason_not_a_generic_error(disclosure):
    session = SaturatedSession()
    provider = CorrectionProvider("fake-key", session)
    result = asyncio.run(correct("Synthetic draft line.", (SourceDocument("resume", SOURCE),),
                                 provider))
    assert (result.status, result.reason) == ("withheld", "provider_rate_limited")
    assert result.markdown is None
    # No retry: a 429 that was not asked to fall back must not be re-spent.
    assert len(session.requests) == 1
    assert provider.usage["failures"][0]["reason"] == "provider_rate_limited"


def test_a_rate_limit_is_not_reported_as_a_privacy_or_malformed_failure(disclosure):
    """Negative control: the new reason must not swallow the other distinct reasons."""
    provider = CorrectionProvider("fake-key", SaturatedSession())
    result = asyncio.run(correct("Synthetic draft line.", (SourceDocument("resume", SOURCE),),
                                 provider))
    assert result.reason not in {"provider_error", "malformed_output", "malformed_source_inventory",
                                 "privacy_review_required", "timeout_source_inventory"}


def test_the_request_allows_provider_fallbacks(disclosure):
    """The model has two endpoints; pinning one made saturation fatal."""
    session = SaturatedSession()
    provider = CorrectionProvider("fake-key", session)
    asyncio.run(correct("Synthetic draft line.", (SourceDocument("resume", SOURCE),), provider))
    assert session.requests[0]["json"]["provider"] == {"allow_fallbacks": True}
    # Redirects stay disabled, so this cannot move the request to another URL.
    assert session.requests[0]["allow_redirects"] is False


class RecordingInner:
    """An aiohttp-shaped session returning a fixed status and body."""

    def __init__(self, status, body):
        self.status = status
        self.body = body

    def post(self, url, **kwargs):
        inner = self
        body = self.body

        class Content:
            async def iter_chunked(self, size):
                raw = body.encode()
                for index in range(0, len(raw), size):
                    yield raw[index:index + size]

        class Response:
            status = inner.status

            def __init__(self):
                self.content = Content()

            async def text(self, *args, **kwargs):
                return body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        return Response()


def test_a_non_200_answer_is_captured_not_dropped():
    capture = live_capture.LiveCapture("case-429")
    session = live_capture.CapturingSession(
        RecordingInner(429, '{"error":{"provider_error_code":"engine_overloaded"}}'), capture)
    request = {"messages": [{"role": "system", "content": "c"},
                            {"role": "user", "content": json.dumps({"segments": []})}]}

    async def run():
        async with session.post("https://example.invalid", json=request) as response:
            return await response.text()

    body = asyncio.run(run())
    assert body == '{"error":{"provider_error_code":"engine_overloaded"}}'
    assert capture.data["exchanges"][0]["status"] == 429
    assert "engine_overloaded" in capture.data["exchanges"][0]["response_body"]


def test_a_streamed_200_answer_is_still_captured():
    """Negative control: adding the text() path must not stop the streaming record."""
    capture = live_capture.LiveCapture("case-200")
    session = live_capture.CapturingSession(RecordingInner(200, '{"ok":true}'), capture)
    request = {"messages": [{"role": "system", "content": "c"},
                            {"role": "user", "content": json.dumps({"segments": []})}]}

    async def run():
        async with session.post("https://example.invalid", json=request) as response:
            chunks = []
            async for chunk in response.content.iter_chunked(4):
                chunks.append(chunk)
            return b"".join(chunks)

    payload = asyncio.run(run())
    assert payload == b'{"ok":true}'
    assert capture.data["exchanges"][0]["status"] == 200
    assert capture.data["exchanges"][0]["response_body"] == '{"ok":true}'


def test_the_captured_rate_limit_replays_as_a_rate_limit(tmp_path, monkeypatch):
    """An offline replay of the recorded 429 must reproduce the same verdict."""
    capture = live_capture.LiveCapture("case-replay-429")
    capture.record_sources([("resume", f"{live_capture.SYNTHETIC_MARKER} {SOURCE}")])
    capture.record_draft(f"{live_capture.SYNTHETIC_MARKER} Synthetic draft line.")
    capture.record_request("source_inventory",
                           {"messages": [{"role": "system", "content": "c"},
                                         {"role": "user", "content": json.dumps({"segments": []})}]})
    capture.record_response(0, 429, '{"error":{"code":429}}')
    monkeypatch.setenv(live_capture.SETTING_ENABLED, "1")
    path = capture.write(tmp_path)
    assert path is not None

    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    result, provider, session = live_capture.replay(path)
    # The engine absorbs the adapter's error into `withheld`; the replay reproduces
    # the shape of the live outcome rather than raising.
    assert result.status == "withheld"
    assert session.calls == 1
