"""A reasoning-only model response must not become a fabricated parse failure.

Measured on both Imaginative diagnostics (diagcons-2c8dd8773b47 and
diagcons-35dbe1af6afd): the Researcher uses ``inception/mercury-2`` with
``response_format={"type": "json_object"}``. The provider returned an empty
``message.content`` and put prose in ``message.reasoning``. The adapter
substituted that prose as the content, so ``json.loads`` failed, the Researcher
set ``fallback_used``, the pipeline reported ``completed_with_fallback`` and the
publication preflight withheld as ``fallback_output`` before correction ran.

For a json_object request the adapter now extracts an embedded JSON object from
the reasoning text when one exists, and otherwise returns an explicit empty
object. The stage can then report an honest "no findings" instead of a parse
failure. Non-JSON callers keep the previous behaviour (reasoning text as-is).
"""
import asyncio
import json

from llm_client_adapter import LLMClientAdapter


class _FakeClient:
    """Returns one scripted OpenRouter-shaped response."""

    def __init__(self, message):
        self.message = message
        self.kwargs = None

    def call_model(self, **kwargs):
        self.kwargs = kwargs
        return {
            "success": True,
            "model_used": "inception/mercury-2",
            "data": {
                "choices": [{"message": self.message}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                          "total_tokens": 30},
            },
        }


def _run(message, response_format=None):
    fake = _FakeClient(message)
    adapter = LLMClientAdapter.__new__(LLMClientAdapter)
    adapter.client = fake
    adapter.usage_stats = {
        "total_prompt_tokens": 0, "total_completion_tokens": 0,
        "total_tokens": 0, "estimated_cost_usd": 0.0, "calls": [],
    }
    return asyncio.run(adapter.call_llm_async(
        system_prompt="s", user_prompt="u", model="inception/mercury-2",
        response_format=response_format))


def test_reasoning_only_prose_becomes_empty_json_for_json_object_requests():
    """The observed failure: prose reasoning must not be handed to json.loads."""
    message = {"content": "", "reasoning": "Let me think about the candidate's "
               "transferable skills. I will consider the job requirements."}
    out = _run(message, {"type": "json_object"})
    assert json.loads(out) == {}, f"expected an empty JSON object, got {out!r}"


def test_embedded_json_in_reasoning_is_extracted():
    """A JSON object inside reasoning text is still usable."""
    payload = {"skill_bridges": [], "domain_vocab": ["trend analysis"]}
    message = {"content": "", "reasoning":
               f"My analysis follows.\n{json.dumps(payload)}\nDone."}
    out = _run(message, {"type": "json_object"})
    assert json.loads(out) == payload


def test_normal_content_is_unchanged_for_json_object_requests():
    """Paired control: a normal JSON response is passed through verbatim."""
    payload = {"skill_bridges": [], "domain_vocab": []}
    message = {"content": json.dumps(payload), "reasoning": "ignored"}
    out = _run(message, {"type": "json_object"})
    assert json.loads(out) == payload


def test_non_json_caller_still_receives_reasoning_text():
    """Paired control: prose callers keep the previous behaviour."""
    message = {"content": "", "reasoning": "A plain prose answer."}
    out = _run(message, None)
    assert out == "A plain prose answer."
