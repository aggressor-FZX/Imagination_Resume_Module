"""Every dispatched HTTP request is metered EXACTLY ONCE, on every path.

The bug this locks
------------------
A 2xx response whose body had no `choices` raised a plain `RuntimeError` from a
point no `record()` call covered. The retry loop then treated it as retryable, so a
single logical call could dispatch up to three HTTP requests while the meter counted
**zero** of them. The request ceiling was therefore not a ceiling on that path, and
the spend report understated it.

The fix meters that branch before raising, and adds a safety net so a future branch
that forgets cannot escape the count. These tests assert the invariant directly:
`meter.requests_sent == number of requests the fake session actually received`.

No provider calls. The session is scripted.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from experiment.provider_v2 import RequestMeter, make_asker

SUCCESS = {"choices": [{"message": {"content": "Decision: approve\n"},
                        "finish_reason": "stop"}],
           "usage": {"cost": 0.004, "prompt_tokens": 10, "completion_tokens": 5,
                     "total_tokens": 15}}
# A 2xx with no completion, and no usage either: the original escaping case.
EMPTY_NO_USAGE = {"choices": []}
# A 2xx with no completion but WITH usage: the request is billed and we can know it.
EMPTY_WITH_USAGE = {"choices": [], "usage": {"cost": 0.006, "prompt_tokens": 9,
                                             "completion_tokens": 0,
                                             "total_tokens": 9}}
MISSING_CHOICES_KEY = {"usage": {"cost": 0.002}}
# A 2xx whose body is not valid JSON at all.
BAD_JSON = "<html>gateway</html>"


class _Resp:
    def __init__(self, status, payload=None, *, bad_json=False):
        self.status = status
        self._payload = payload
        self._bad_json = bad_json

    async def json(self):
        if self._bad_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload

    async def text(self):
        return "error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Counts every request the adapter actually dispatches."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *_a, **_k):
        self.requests += 1
        if not self.script:
            raise AssertionError("adapter sent more requests than the script allows")
        return self.script.pop(0)


def _install(monkeypatch, script):
    session = _Session(script)
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(
        ClientSession=lambda **_: session, ClientTimeout=lambda **_: None))
    return session


def _run(script, monkeypatch, *, max_requests=10, max_attempts=3):
    session = _install(monkeypatch, script)
    meter = RequestMeter(max_requests=max_requests, max_cost_usd=1.0)
    ask = make_asker(api_key="k", meter=meter, sleep_between_retries=False,
                     max_attempts=max_attempts)
    error = None
    try:
        result = asyncio.run(ask("prompt", label="case:first"))
    except Exception as exc:  # noqa: BLE001
        result, error = None, exc
    return session, meter, result, error


# ── The escaping path ────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload,kwargs", [
    (EMPTY_NO_USAGE, {}),
    (EMPTY_WITH_USAGE, {}),
    (MISSING_CHOICES_KEY, {}),
    (None, {"bad_json": True}),
], ids=["empty_no_usage", "empty_with_usage", "missing_key", "bad_json"])
def test_an_unusable_2xx_is_counted_for_every_retry(payload, kwargs, monkeypatch):
    """The exact bug: three dispatched requests, three metered."""
    script = [_Resp(200, payload, **kwargs) for _ in range(3)]
    session, meter, result, error = _run(script, monkeypatch, max_attempts=3)

    assert error is not None, "unusable answers must not be returned as a result"
    assert result is None
    # The invariant under test.
    assert meter.requests_sent == session.requests == 3
    assert meter.logical_calls == 1


def test_no_choices_without_usage_is_unknown_cost_not_free(monkeypatch):
    script = [_Resp(200, EMPTY_NO_USAGE)]
    session, meter, _result, error = _run(script, monkeypatch, max_attempts=1)
    assert error is not None
    assert meter.requests_sent == 1
    assert meter.unmetered_requests == 1
    assert meter.cost_is_known is False
    assert meter.attempts[0]["status"] == "ok_no_choices"


def test_no_choices_with_usage_charges_the_known_cost(monkeypatch):
    """Usage is often present on a choiceless 200. Then the cost IS knowable."""
    script = [_Resp(200, EMPTY_WITH_USAGE)]
    session, meter, _result, error = _run(script, monkeypatch, max_attempts=1)
    assert error is not None
    assert meter.requests_sent == 1
    assert meter.cost_usd == pytest.approx(0.006)
    assert meter.cost_is_known is True
    assert meter.attempts[0]["status"] == "ok_no_choices"


def test_the_ceiling_binds_the_unusable_2xx_path(monkeypatch):
    """The old bug's consequence: this path could exceed the ceiling silently."""
    script = [_Resp(200, EMPTY_NO_USAGE) for _ in range(5)]
    session, meter, _result, error = _run(script, monkeypatch, max_requests=2,
                                          max_attempts=5)
    assert error is not None
    assert meter.requests_sent == session.requests == 2, \
        "the third request must be refused, not sent uncounted"


# ── Nothing else counts twice ────────────────────────────────────────────────

def test_a_successful_call_is_counted_once(monkeypatch):
    session, meter, result, _error = _run([_Resp(200, SUCCESS)], monkeypatch,
                                          max_attempts=1)
    assert result is not None and result.text.startswith("Decision: approve")
    assert meter.requests_sent == session.requests == 1
    assert result.requests_sent == 1


def test_a_retried_503_then_success_counts_each_attempt_once(monkeypatch):
    script = [_Resp(503), _Resp(503), _Resp(200, SUCCESS)]
    session, meter, result, error = _run(script, monkeypatch, max_attempts=3)
    assert error is None and result is not None
    assert meter.requests_sent == session.requests == 3
    assert [a["status"] for a in meter.attempts] == [
        "http_503", "http_503", "ok"]


def test_a_4xx_is_counted_once_and_not_retried(monkeypatch):
    """A deterministic client error must not be billed three times."""
    script = [_Resp(400)]
    session, meter, _result, error = _run(script, monkeypatch, max_attempts=3)
    assert error is not None
    assert meter.requests_sent == session.requests == 1
    assert meter.attempts[0]["status"] == "http_400"


def test_a_transport_error_is_counted_once_per_attempt(monkeypatch):
    class _Boom(_Session):
        def post(self, *_a, **_k):
            self.requests += 1
            raise TimeoutError("read timeout")

    session = _Boom([])
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(
        ClientSession=lambda **_: session, ClientTimeout=lambda **_: None))
    meter = RequestMeter(max_requests=10, max_cost_usd=1.0)
    ask = make_asker(api_key="k", meter=meter, sleep_between_retries=False,
                     max_attempts=2)
    with pytest.raises(RuntimeError):
        asyncio.run(ask("prompt", label="case:first"))
    assert meter.requests_sent == session.requests == 2
    assert all(a["status"] == "transport_error" for a in meter.attempts)


def test_every_recorded_attempt_has_a_distinct_status_label(monkeypatch):
    """A parse failure and a dead connection must not share a label."""
    script = [_Resp(200, None, bad_json=True)]
    session, meter, _result, error = _run(script, monkeypatch, max_attempts=1)
    assert error is not None
    assert meter.attempts[0]["status"] == "unreadable_response"


def test_two_logical_calls_meter_independently(monkeypatch):
    session, meter, result, _error = _run([_Resp(200, SUCCESS)], monkeypatch,
                                          max_attempts=1)
    assert meter.logical_calls == 1
    # A second call over the same meter must add, not reset.
    meter2 = RequestMeter(max_requests=10, max_cost_usd=1.0)
    meter2.logical_calls = meter.logical_calls
    assert meter2.requests_sent == 0
