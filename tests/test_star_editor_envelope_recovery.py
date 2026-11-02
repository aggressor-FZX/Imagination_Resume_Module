"""The StarEditor envelope failures, and the transport setting that caused them.

Observed live on 2026-11-02 19:41 and twice on 2026-11-01: the StarEditor's answer
would not parse, so it returned an empty result, the pipeline marked itself
``fallback_output``, and the publication gate correctly refused to publish. That is a
delivery failure, and it happened *before* source correction, so no capture could
describe it.

The structural cause was found by reading the transport rather than guessing: the
adapter accepted ``response_format={"type": "json_object"}`` from every stage and
never placed it on the wire. The payload was a fixed five-key dict. So every stage
that asked for strict JSON silently ran unstructured, and the editor — the one stage
whose whole output is a JSON envelope — took the visible damage.

These tests pin the forwarding and the narrow, verifiable recovery for the specific
failure shape that was observed.
"""
import json

import pytest

from openrouter_safe_client import OpenRouterSafeClient
from stages.star_editor import StarEditor


class _Response:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _client(monkeypatch, payload):
    """A client whose HTTP layer is replaced, so the payload can be inspected."""
    client = OpenRouterSafeClient("test-key")
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False, **kwargs):
        sent.update(json or {})
        return _Response(payload)

    monkeypatch.setattr("openrouter_safe_client.requests.post", fake_post)
    monkeypatch.setattr(client, "_is_model_available", lambda model: True)
    return client, sent


def test_json_mode_is_forwarded_to_the_provider(monkeypatch):
    """The setting must reach the wire, not stop at the adapter."""
    client, sent = _client(monkeypatch, {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    result = client.call_model(
        messages=[{"role": "user", "content": "hi"}], model="some/model",
        response_format={"type": "json_object"})
    assert result["success"] is True
    assert sent["response_format"] == {"type": "json_object"}


def test_omitting_json_mode_does_not_invent_it(monkeypatch):
    """Negative control: a stage that did not ask for JSON must not get it."""
    client, sent = _client(monkeypatch, {
        "choices": [{"message": {"content": "plain text"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    client.call_model(messages=[{"role": "user", "content": "hi"}], model="some/model")
    assert "response_format" not in sent


def _editor():
    return StarEditor(llm_client=None)


def test_a_complete_markdown_value_in_a_broken_envelope_is_recovered():
    """The observed shape: the body is intact, only the envelope is malformed."""
    markdown = "## Professional Experience\n**Acme** -- Example City, WA\n- Did a thing."
    encoded = json.dumps(markdown)
    # Cut the envelope after the closed string value, so it cannot parse as JSON.
    broken = '{"final_markdown": ' + encoded + ', "final_plain_text": "x"'
    result = _editor()._parse_response(broken)
    assert result["parse_error"] == "truncated_json"
    assert result["fallback_used"] is False
    assert "## Professional Experience" in result["final_markdown"]
    assert "**Acme**" in result["final_markdown"]


def test_a_body_cut_mid_sentence_is_not_published():
    """Negative control: recovery must reject text truncated inside the value.

    Publishing a half-written resume is worse than withholding it, because the
    missing tail looks like the candidate simply had fewer achievements.
    """
    markdown = "## Professional Experience\n**Acme** -- Example City, WA\n- Did a thing"
    encoded = json.dumps(markdown)
    broken = '{"final_markdown": ' + encoded[:-6]  # no closing quote at all
    result = _editor()._parse_response(broken)
    assert result["final_markdown"] == ""
    assert result["fallback_used"] is True


def test_escaped_quotes_do_not_end_the_value_early():
    markdown = '## Summary\n- Shipped a "zero downtime" migration.\n## Education\n**School**'
    encoded = json.dumps(markdown)
    broken = '{"final_markdown": ' + encoded + ', "editorial_notes": "x"'
    result = _editor()._parse_response(broken)
    assert result["parse_error"] == "truncated_json"
    assert '"zero downtime"' in result["final_markdown"]


def test_recovery_requires_actual_resume_structure():
    """Negative control: an arbitrary string in that key is not a resume."""
    broken = '{"final_markdown": "just some words", "final_plain_text": "x"'
    result = _editor()._parse_response(broken)
    assert result["final_markdown"] == ""
    assert result["fallback_used"] is True


def test_a_valid_envelope_is_unaffected():
    """Negative control: the recovery path must not intercept well-formed JSON."""
    markdown = "## Professional Experience\n**Acme** -- Example City, WA\n- Did a thing."
    valid = json.dumps({"final_markdown": markdown, "final_plain_text": "x",
                        "editorial_notes": "ok"})
    result = _editor()._parse_response(valid)
    assert "parse_error" not in result
    assert "**Acme**" in result["final_markdown"]


def test_recovered_markdown_still_passes_the_metric_verifier():
    """Recovery must not bypass verification; the content is still checked."""
    import tier_a_verifier

    source = "## Professional Experience\n**Acme** -- Example City, WA\n- Improved yield by 30%."
    markdown = "## Professional Experience\n**Acme** -- Example City, WA\n- Improved yield by 30%."
    broken = '{"final_markdown": ' + json.dumps(markdown) + ', "final_plain_text": "x"'
    result = _editor()._parse_response(broken)
    verdict = tier_a_verifier.verify_metrics(result["final_markdown"], source)
    assert verdict["verdict"] == "PASS (tier A only)"
