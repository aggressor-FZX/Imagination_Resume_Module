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
    def __init__(self, status, payload=None):
        self.status = status
        self._payload = payload or {}

    async def json(self):
        return self._payload

    async def text(self):
        return "err"

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
