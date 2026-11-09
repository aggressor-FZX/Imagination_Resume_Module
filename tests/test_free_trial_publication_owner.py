"""[free-trial-owner] A paid free-trial resume must not be discarded for being logged out.

Measured 2026-11-08: 1 of 20 correction runs withheld as ``receipt_unavailable``.
The correction had APPROVED -- the resume was rewritten, reviewed and passed every
gate -- and was then thrown away because ``app.py`` withheld whenever the account
owner header was absent. The orchestrator only sends that header when ``req.user``
exists (``server.cjs``), so a free-trial visitor has no account owner and every
such run was discarded.

That reason is also outside the engine's ``allowed_reasons`` set, and it is
produced after ``correction.status == "approved"``: the approval genuinely
happened, and the user was told their resume "could not be verified".

These tests pin the three properties that make the fix safe:

1. a signed-in account still wins, and keeps its own namespace;
2. a valid visitor identity becomes an owner, namespaced so it can never collide
   with an account id;
3. a malformed, absent, or browser-fallback visitor id yields NO owner, so the
   approval is withheld rather than bound to something untrustworthy.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app as app_module


def _client(headers):
    """A request whose headers are exactly what the orchestrator would forward."""
    probe = FastAPI()

    @probe.get("/probe")
    async def read():  # pragma: no cover - exercised through TestClient
        owner, is_account = app_module.resolve_publication_owner(_REQUEST[-1])
        return {"owner": owner, "is_account": is_account}

    _REQUEST.clear()
    inner = TestClient(probe)

    @probe.middleware("http")
    async def capture(request, call_next):
        _REQUEST.append(request)
        return await call_next(request)

    return inner, headers


_REQUEST = []


@pytest.mark.parametrize("headers,expected,is_account", [
    # A signed-in account always wins, even if a device id is also present.
    ({"x-cogito-owner-id": "42", "x-cogito-visitor-id": "AbCdEfGhIjK"},
     "42", True),
    # A free-trial visitor becomes a namespaced owner. Without this the approval
    # was withheld and the resume discarded.
    ({"x-cogito-visitor-id": "AbCdEfGhIjK"}, "visitor:AbCdEfGhIjK", False),
])
def test_resolved_owner_and_kind(headers, expected, is_account):
    client, _ = _client(headers)
    body = client.get("/probe", headers=headers).json()
    assert body["owner"] == expected
    assert body["is_account"] is is_account


@pytest.mark.parametrize("headers", [
    {},                                                  # no identity at all
    {"x-cogito-visitor-id": "short"},                     # below the minimum
    {"x-cogito-visitor-id": "fallback_abcdefghijk"},      # browser fallback id
    {"x-cogito-visitor-id": "has spaces in it"},          # not a device id
    {"x-cogito-visitor-id": "A" * 200},                   # unreasonable length
])
def test_untrustworthy_identity_yields_no_owner(headers):
    """No owner means the approval is withheld -- which is correct here.

    Binding an approval to a short, fallback, or malformed id would silently
    weaken the binding it exists to provide, so refusing is the safe outcome.
    """
    client, _ = _client(headers)
    body = client.get("/probe", headers=headers).json()
    assert body["owner"] is None
    assert body["is_account"] is False


def test_visitor_owner_can_never_collide_with_an_account_id():
    """The namespace is load-bearing.

    An account id is a verified login; a device id is a clearable cookie. If the
    two could produce the same receipt owner, a visitor-bound approval would be
    indistinguishable from an account-bound one at export.
    """
    assert app_module.VISITOR_OWNER_PREFIX.endswith(":")
    # A numeric account id can never be produced by the visitor path, because the
    # prefix is always applied.
    visitor_owner = app_module.VISITOR_OWNER_PREFIX + "42"
    assert visitor_owner != "42"
    assert not visitor_owner.isdigit()


@pytest.mark.parametrize("reason,expect_retry", [
    # Our failures: the document was never the problem.
    ("timeout_initial_review", True),
    ("timeout_repair", True),
    ("malformed_output", True),
    ("no_op_repair", True),
    ("provider_unavailable", True),
    # The document could not be substantiated.
    ("source_facts_missing", False),
    ("source_completeness_uncertain", False),
    ("attribution_conflict", False),
    ("unresolved_claims", False),
])
def test_review_message_never_blames_the_resume_for_our_failures(reason, expect_retry):
    """The one hardcoded sentence used to cover every reason.

    A 74-second provider timeout and a genuinely unplaceable fact produced
    identical text that read as a judgement on the candidate's document. The
    message is now chosen by whose failure it was, and ``retryable`` is its
    machine-readable twin so a client can offer a retry only when it would help.
    """
    message = app_module._review_message_for(reason)
    assert message is not None and message.strip()
    assert message != app_module._CONTENT_MESSAGE or not expect_retry
    assert (reason not in app_module._CONTENT_REASONS) is expect_retry
    if expect_retry:
        assert "not rejected" in message
    else:
        assert "could not confirm" in message


def test_receipt_unavailable_is_actionable_not_retryable():
    """Retrying cannot fix it, and it is not a judgement on the document.

    A receipt that could not be bound to an owner fails identically every time the
    visitor stays logged out. "Please try again" would be as wrong here as "your
    resume could not be verified" -- both send the user somewhere useless.
    """
    message = app_module._review_message_for("receipt_unavailable")
    assert message == app_module._ACTIONABLE_MESSAGE
    assert "sign in" in message.lower()
    assert "receipt_unavailable" in app_module._ACTIONABLE_REASONS
    assert "receipt_unavailable" not in app_module._CONTENT_REASONS
    # Neither class: it is fixable, but not by repeating the request.
    withheld = app_module._withhold_correction(
        reason="receipt_unavailable", pipeline_status="completed",
        metrics={"calls": [], "failures": [], "total_prompt_tokens": 0,
                 "total_completion_tokens": 0, "total_tokens": 0,
                 "estimated_cost_usd": 0.0},
        start_time=0.0)
    assert withheld.retryable is False
    assert withheld.review_message == app_module._ACTIONABLE_MESSAGE


def test_an_unknown_reason_is_treated_as_retryable():
    """Safe default: inviting a retry costs one request, while telling someone
    to re-check a resume that was actually fine costs them work."""
    assert app_module._review_message_for("something_new") == app_module._RETRY_MESSAGE
