"""Provider pinning must actually reach the request body, and be auditable.

Why this matters for a model comparison
---------------------------------------
MiMo-V2.6-Flash is served by EIGHT endpoints: fp4, bf16 and six fp8. Without
`provider: {only: [...]}` OpenRouter may route each request independently, so two
calls of the same prompt are not necessarily the same model instance — and an fp4
routing would be materially weaker than the fp8 the result is attributed to.

Two things are tested:

1. the pin is present in the outbound body (not merely stored on the asker);
2. what the provider REPORTS is recorded, so a pin that did not hold is visible
   instead of assumed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from experiment.provider_v2 import RequestMeter, make_asker

SUCCESS = {"choices": [{"message": {"content": "Decision: approve\n"},
                        "finish_reason": "stop"}],
           "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 5,
                     "total_tokens": 15},
           "provider": "Cohere", "model": "xiaomi/mimo-v2.6-flash"}


class _Resp:
    def __init__(self, status, payload=None, *, detail="err"):
        self.status = status
        self._payload = payload or {}
        self._detail = detail

    async def json(self):
        return self._payload

    async def text(self):
        return self._detail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Captures the JSON body of every request."""

    def __init__(self, script):
        self.script = list(script)
        self.bodies: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, _url, *, headers=None, json=None):
        self.bodies.append(json)
        return self.script.pop(0)


def _install(monkeypatch, script):
    session = _Session(script)
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(
        ClientSession=lambda **_: session, ClientTimeout=lambda **_: None))
    return session


def _run(monkeypatch, *, provider_only=None, payload=None, model="m"):
    session = _install(monkeypatch, [_Resp(200, payload or SUCCESS)])
    meter = RequestMeter(max_requests=5, max_cost_usd=1.0)
    ask = make_asker(api_key="k", model=model, meter=meter,
                     provider_only=provider_only, sleep_between_retries=False,
                     max_attempts=1)
    result = asyncio.run(ask("prompt", label="x"))
    return session, meter, result


# ── The pin reaches the wire ────────────────────────────────────────────────

def test_the_pin_is_sent_in_the_request_body(monkeypatch):
    session, _m, _r = _run(monkeypatch, provider_only=["cohere/fp8"],
                           model="xiaomi/mimo-v2.6-flash")
    body = session.bodies[0]
    assert body["provider"] == {"only": ["cohere/fp8"]}
    assert body["model"] == "xiaomi/mimo-v2.6-flash"


def test_no_provider_field_when_unpinned(monkeypatch):
    """An unpinned model must not gain an empty provider object."""
    session, _m, _r = _run(monkeypatch, provider_only=None)
    assert "provider" not in session.bodies[0]


def test_multiple_pins_are_preserved_in_order(monkeypatch):
    session, _m, _r = _run(monkeypatch, provider_only=["cohere/fp8", "makora/fp8"])
    assert session.bodies[0]["provider"]["only"] == ["cohere/fp8", "makora/fp8"]


# ── What was pinned vs what served ──────────────────────────────────────────

def test_the_reported_provider_is_recorded(monkeypatch):
    _s, _m, result = _run(monkeypatch, provider_only=["cohere/fp8"])
    assert result.provider_pinned == ["cohere/fp8"]
    assert result.provider_reported == "Cohere"


def test_a_pin_mismatch_is_visible_not_hidden(monkeypatch):
    """If OpenRouter routes elsewhere, the record must show it.

    The pin is an instruction; only the response says what happened. This test
    forces a different provider in the payload and asserts both values survive.
    """
    payload = {**SUCCESS, "provider": "DeepInfra"}
    _s, _m, result = _run(monkeypatch, provider_only=["cohere/fp8"], payload=payload)
    assert result.provider_pinned == ["cohere/fp8"]
    assert result.provider_reported == "DeepInfra"
    assert result.provider_reported != "Cohere"


def test_pinning_does_not_disturb_metering(monkeypatch):
    _s, meter, result = _run(monkeypatch, provider_only=["cohere/fp8"])
    assert meter.requests_sent == 1
    assert result.requests_sent == 1
    assert meter.cost_usd == pytest.approx(0.001)


def test_to_dict_carries_the_routing_fields(monkeypatch):
    _s, _m, result = _run(monkeypatch, provider_only=["cohere/fp8"])
    d = result.to_dict()
    assert d["provider_pinned"] == ["cohere/fp8"]
    assert d["provider_reported"] == "Cohere"


# ── The screen declares the pin ─────────────────────────────────────────────

def test_the_screen_pins_mimo_to_the_requested_endpoint():
    from experiment.model_screen import PROVIDER_FALLBACKS, PROVIDER_PINS
    assert PROVIDER_PINS["xiaomi/mimo-v2.6-flash"] == ["cohere/fp8"]
    assert PROVIDER_FALLBACKS["xiaomi/mimo-v2.6-flash"] == ["makora/fp8"]


def test_the_plan_declares_pins_and_fallbacks():
    from experiment.model_screen import MODELS, budget_for
    plan = budget_for(MODELS)
    assert plan["provider_pins"] == {"xiaomi/mimo-v2.6-flash": ["cohere/fp8"]}
    assert plan["provider_fallbacks"] == {"xiaomi/mimo-v2.6-flash": ["makora/fp8"]}
    # Unpinned models must not appear in the pin map at all.
    assert set(plan["provider_pins"]) == {"xiaomi/mimo-v2.6-flash"}


# ══ Provider fallback: cohere first, makora only if cohere cannot serve ══════

class _ScriptedSession:
    """Serves a per-pin sequence, so the chain can be exercised deterministically."""

    def __init__(self, by_pin: dict[str, list]):
        self.by_pin = {k: list(v) for k, v in by_pin.items()}
        self.bodies: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, _url, *, headers=None, json=None):
        self.bodies.append(json)
        pin = ",".join(json.get("provider", {}).get("only", []))
        queue = self.by_pin.get(pin)
        if not queue:
            raise AssertionError(f"no scripted response for pin {pin!r}")
        return queue.pop(0)


def _install_scripted(monkeypatch, by_pin):
    session = _ScriptedSession(by_pin)
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(
        ClientSession=lambda **_: session, ClientTimeout=lambda **_: None))
    return session


def _run_chain(monkeypatch, by_pin, *, primary=("cohere/fp8",),
               fallback=("makora/fp8",)):
    session = _install_scripted(monkeypatch, by_pin)
    meter = RequestMeter(max_requests=10, max_cost_usd=1.0)
    ask = make_asker(api_key="k", model="xiaomi/mimo-v2.6-flash", meter=meter,
                     provider_only=list(primary),
                     provider_fallback=list(fallback),
                     sleep_between_retries=False, max_attempts=1)
    try:
        result = asyncio.run(ask("prompt", label="c"))
        error = None
    except Exception as exc:  # noqa: BLE001
        result, error = None, exc
    return session, meter, result, error


def test_cohere_serves_the_call_and_makora_is_never_tried(monkeypatch):
    ok = {**SUCCESS, "provider": "Cohere"}
    session, _m, result, error = _run_chain(
        monkeypatch, {"cohere/fp8": [_Resp(200, ok)], "makora/fp8": []})
    assert error is None
    assert result.provider_reported == "Cohere"
    assert [b.get("provider", {}).get("only") for b in session.bodies] == \
        [["cohere/fp8"]], "the fallback must not be contacted at all"


def test_a_routing_failure_on_cohere_advances_to_makora(monkeypatch):
    """`no endpoints found` means cohere cannot serve it, so try the other one.

    The DETAIL text carries the routing phrase, because the adapter raises
    `http_<code>: <detail>` and the detector reads that message.
    """
    mak = {**SUCCESS, "provider": "Makora"}
    session, _m, result, error = _run_chain(monkeypatch, {
        "cohere/fp8": [_Resp(404, detail="no endpoints found for cohere/fp8")],
        "makora/fp8": [_Resp(200, mak)],
    })
    assert error is None, error
    assert result is not None
    assert result.provider_reported == "Makora"
    # Both providers were actually contacted, in order, and the pin is recorded
    # as the one that SERVED the call, not the one we hoped for.
    assert [b.get("provider", {}).get("only") for b in session.bodies] == \
        [["cohere/fp8"], ["makora/fp8"]]
    assert result.provider_pinned == ["makora/fp8"]


def test_a_bare_404_does_not_advance_the_chain(monkeypatch):
    """A 404 with no routing phrase is an ordinary failure, not a routing failure."""
    _s, _m, result, error = _run_chain(monkeypatch, {
        "cohere/fp8": [_Resp(404, detail="not found")],
        "makora/fp8": [_Resp(200, {**SUCCESS, "provider": "Makora"})],
    })
    assert error is not None, "must surface, not silently switch endpoint"
    assert result is None


def test_an_auth_error_does_not_advance_the_chain(monkeypatch):
    """A key problem must be reported, not masked by asking another provider."""
    _s, _m, result, error = _run_chain(monkeypatch, {
        "cohere/fp8": [_Resp(401, detail="no auth credentials found")],
        "makora/fp8": [_Resp(200, {**SUCCESS, "provider": "Makora"})],
    })
    assert error is not None and result is None
    assert "401" in str(error)


def test_a_bad_answer_does_not_trigger_the_fallback(monkeypatch):
    """A refusal is still a review. Re-asking another endpoint would be best-of-two."""
    refusal = {"choices": [{"message": {"content": "Decision: revise\n\nIssues:\n- none\n"},
                            "finish_reason": "stop"}],
               "usage": {"cost": 0.001}, "provider": "Cohere"}
    session, _m, result, error = _run_chain(
        monkeypatch, {"cohere/fp8": [_Resp(200, refusal)], "makora/fp8": []})
    assert error is None
    assert result.provider_reported == "Cohere"
    assert len(session.bodies) == 1, "no second provider may be asked"
    assert result.text.startswith("Decision: revise")


def test_routing_failure_detector_is_narrow():
    """Auth, rate-limit and server errors must NOT be treated as routing failures."""
    from experiment.provider_v2 import _is_routing_failure
    for msg in ("http_400: no endpoints found", "http_404: no endpoints found",
                "provider returned error", "model not available"):
        assert _is_routing_failure(msg) is True, msg
    for msg in ("http_401: no auth credentials found", "http_429: rate limited",
                "http_500: upstream exploded", "http_502: bad gateway",
                "http_400: invalid request json"):
        assert _is_routing_failure(msg) is False, msg


def test_the_screen_requests_the_cohere_then_makora_chain():
    from experiment.model_screen import MODELS, PROVIDER_FALLBACKS, PROVIDER_PINS
    assert "xiaomi/mimo-v2.6-flash" in MODELS
    assert PROVIDER_PINS["xiaomi/mimo-v2.6-flash"] == ["cohere/fp8"]
    assert PROVIDER_FALLBACKS["xiaomi/mimo-v2.6-flash"] == ["makora/fp8"]
