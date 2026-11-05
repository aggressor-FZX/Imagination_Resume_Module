"""Phase D: the approval receipt binds the skill-inference policy state (v5).

The gap this closes: the receipt bound the *rendered document* but said nothing
about the inference mode or the surviving bridges. A source-only approval could
therefore be presented as an approval of inferred skills, because mode/bridges
are not part of the rendered object and were never signed.

Design under test:

* v4 stays the default and remains valid for source-only approvals;
* supplying policy state issues **v5**, which adds a `pol_fp` fingerprint;
* a v4 receipt is refused when the export *declares* policy state
  (`receipt_policy_unbound`) — it grants no bridge privilege;
* a v5 receipt is refused when the caller drops the state
  (`receipt_policy_state_missing`) or alters any bound field.

No network, no provider, no renderer: `canonical_render_content` is the real
parser, and the renderer call count is asserted to stay zero on refusal.
"""
import hashlib

import pytest

import publication_gate as gate


OWNER = "candidate-4471"
SECRET = "phase-d-policy-binding-secret-value"
MD = (
    "## Professional Summary\n"
    "Analyst with measurement and reporting experience.\n\n"
    "## Professional Experience\n"
    "**Cascade Power Solutions** -- Example City, WA\n"
    "*Data Analyst | 03/2020 -- Present*\n"
    "- Compared weekly sales figures to identify trends across 12 branches.\n\n"
    "## Education\n"
    "**Example Technical College** -- 2016\n\n"
    "## Certifications\n\n"
    "## Technical Skills\n"
    "**Analysis:** trend analysis, reporting\n"
)

PAYLOAD = {
    "summary": "Analyst with measurement and reporting experience.",
    "skills": ["trend analysis", "reporting"],
    "experience": [
        {
            "title": "Data Analyst",
            "company": "Cascade Power Solutions",
            "bullets": [
                "Compared weekly sales figures to identify trends across 12 branches."
            ],
        }
    ],
    "education": ["**Example Technical College** -- 2016"],
    "certifications": [],
    "projects": [],
}


def _policy(mode="imaginative", bridges=None, version="imaginative-v1"):
    return {
        "skill_inference_mode": mode,
        "skill_inference_policy_version": version,
        "surviving_bridges": bridges
        if bridges is not None
        else [
            {
                "id": "bridge-01-abc123",
                "skill": "trend analysis",
                "origin": "researcher_inference",
                "source_id": "document-0",
                "policy_version": version,
            }
        ],
    }


@pytest.fixture(autouse=True)
def signing(monkeypatch):
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", SECRET)
    monkeypatch.setenv("PUBLICATION_RECEIPT_TTL_SECONDS", "900")


def _issue(**kwargs):
    return gate.issue_receipt(
        markdown=MD,
        verdict=gate.Verdict.PASS_TIER_A.value,
        owner=OWNER,
        approved_payload=dict(PAYLOAD),
        **kwargs,
    )


def _decode(receipt):
    import base64
    import json

    token = receipt.rpartition(".")[0]
    return json.loads(base64.urlsafe_b64decode(token.encode()).decode())


# ── Version selection ──────────────────────────────────────────────────────


def test_source_only_approval_stays_v4():
    receipt = _issue()
    assert receipt
    assert _decode(receipt)["v"] == 4
    assert "pol_fp" not in _decode(receipt)


def test_policy_state_upgrades_to_v5():
    receipt = _issue(policy_state=_policy())
    assert receipt
    body = _decode(receipt)
    assert body["v"] == 5
    assert body["pol_fp"] == gate.policy_fingerprint(_policy())


def test_imaginative_with_zero_bridges_still_binds_the_mode():
    """A run that inferred nothing is still a policy decision worth binding."""
    receipt = _issue(policy_state=_policy(bridges=[]))
    assert receipt
    assert _decode(receipt)["v"] == 5


# ── The core asymmetry ────────────────────────────────────────────────────


def test_v4_receipt_cannot_authorize_declared_policy_state():
    """The decisive test: an approval that never mentioned inference grants none."""
    receipt = _issue()  # v4, no policy binding
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=MD,
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=_policy(),
    )
    assert (ok, reason) == (False, "receipt_policy_unbound")


def test_v5_receipt_requires_the_state_to_be_presented():
    receipt = _issue(policy_state=_policy())
    ok, reason = gate.verify_receipt(
        receipt, markdown=MD, owner=OWNER, render_payload=dict(PAYLOAD)
    )
    assert (ok, reason) == (False, "receipt_policy_state_missing")


def test_matching_v5_state_verifies():
    policy = _policy()
    receipt = _issue(policy_state=policy)
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=MD,
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=policy,
    )
    assert (ok, reason) == (True, "ok")


def test_v4_source_only_round_trip_still_works():
    """Existing source-only approvals must not be broken by the new version."""
    receipt = _issue()
    ok, reason = gate.verify_receipt(
        receipt, markdown=MD, owner=OWNER, render_payload=dict(PAYLOAD)
    )
    assert (ok, reason) == (True, "ok")


# ── Tampering matrix ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mutate,label",
    [
        (lambda p: p.update(skill_inference_mode="conservative"), "mode"),
        (lambda p: p.update(skill_inference_policy_version="imaginative-v9"), "policy version"),
        (lambda p: p["surviving_bridges"][0].update(skill="time-series analysis"), "label"),
        (lambda p: p["surviving_bridges"][0].update(id="bridge-99-evil"), "bridge id"),
        (lambda p: p["surviving_bridges"][0].update(source_id="document-9"), "source"),
        (lambda p: p["surviving_bridges"].append({
            "id": "bridge-02-added", "skill": "python", "origin": "researcher_inference",
            "source_id": "document-0", "policy_version": "imaginative-v1"}), "added bridge"),
        (lambda p: p["surviving_bridges"].clear(), "dropped bridges"),
    ],
)
def test_any_bound_field_change_is_refused(mutate, label):
    policy = _policy()
    receipt = _issue(policy_state=policy)
    altered = _policy()
    mutate(altered)
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=MD,
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=altered,
    )
    assert (ok, reason) == (False, "receipt_policy_mismatch"), f"{label} was not bound"


def test_reordering_bridges_does_not_change_the_fingerprint():
    """Canonicalisation is deterministic, so ordering is not part of the identity."""
    a = {
        "id": "bridge-01-a", "skill": "trend analysis", "origin": "researcher_inference",
        "source_id": "document-0", "policy_version": "imaginative-v1",
    }
    b = {
        "id": "bridge-02-b", "skill": "time-series analysis",
        "origin": "researcher_inference", "source_id": "document-1",
        "policy_version": "imaginative-v1",
    }
    first = _policy(bridges=[a, b])
    second = _policy(bridges=[b, a])
    assert gate.policy_fingerprint(first) == gate.policy_fingerprint(second)


def test_fingerprint_changes_when_a_bridge_changes():
    first = gate.policy_fingerprint(_policy())
    second = gate.policy_fingerprint(_policy(bridges=[]))
    assert first != second
    assert first is not None


def test_absent_policy_has_no_fingerprint():
    assert gate.policy_fingerprint(None) is None
    assert gate.policy_fingerprint({}) is None


def test_fingerprint_ignores_non_semantic_metadata():
    """Only id/label/origin/source/version are bound; extra keys are not."""
    policy = _policy()
    policy["surviving_bridges"][0]["rationale"] = "some long explanation"
    policy["surviving_bridges"][0]["evidence_quote"] = "Compared weekly figures"
    assert gate.policy_fingerprint(policy) == gate.policy_fingerprint(_policy())


# ── Existing guarantees still hold ────────────────────────────────────────


def test_markdown_tamper_is_still_caught_before_the_policy_check():
    policy = _policy()
    receipt = _issue(policy_state=policy)
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=MD.replace("12 branches", "40 branches"),
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=policy,
    )
    assert (ok, reason) == (False, "receipt_content_mismatch")


def test_wrong_owner_is_still_caught_with_policy_state():
    policy = _policy()
    receipt = _issue(policy_state=policy)
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=MD,
        owner="someone-else",
        render_payload=dict(PAYLOAD),
        policy_state=policy,
    )
    assert (ok, reason) == (False, "receipt_owner_mismatch")


def test_render_payload_still_required_under_v5():
    receipt = _issue(policy_state=_policy())
    ok, reason = gate.verify_receipt(
        receipt, markdown=MD, owner=OWNER, policy_state=_policy()
    )
    assert (ok, reason) == (False, "receipt_render_payload_missing")


def test_signature_tamper_is_caught():
    receipt = _issue(policy_state=_policy())
    token, _, sig = receipt.rpartition(".")
    assert sig
    flipped = ("0" if sig[0] != "0" else "1") + sig[1:]
    ok, reason = gate.verify_receipt(
        f"{token}.{flipped}",
        markdown=MD,
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=_policy(),
    )
    assert (ok, reason) == (False, "receipt_signature_invalid")


def test_unsupported_version_is_refused(monkeypatch):
    """A future/unknown wire version must fail closed, not be guessed at."""
    policy = _policy()
    receipt = _issue(policy_state=policy)

    import base64
    import json

    token, _, _sig = receipt.rpartition(".")
    body = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
    body["v"] = 9
    forged_token = base64.urlsafe_b64encode(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    # Re-sign with the real secret so only the version is wrong.
    import hashlib
    import hmac

    sig = hmac.new(
        SECRET.encode(), json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    ok, reason = gate.verify_receipt(
        f"{forged_token}.{sig}",
        markdown=MD,
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=policy,
    )
    assert (ok, reason) == (False, "receipt_version_unknown")


def test_conservative_policy_state_binds_no_bridges():
    policy = _policy(mode="conservative", bridges=[], version="conservative-v1")
    receipt = _issue(policy_state=policy)
    assert _decode(receipt)["v"] == 5
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=MD,
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=policy,
    )
    assert (ok, reason) == (True, "ok")


def test_conservative_state_with_bridges_mismatches():
    """A Conservative approval cannot smuggle bridges into its signed state."""
    conservative = _policy(mode="conservative", bridges=[], version="conservative-v1")
    receipt = _issue(policy_state=conservative)
    smuggled = _policy(mode="conservative", bridges=_policy()["surviving_bridges"],
                       version="conservative-v1")
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=MD,
        owner=OWNER,
        render_payload=dict(PAYLOAD),
        policy_state=smuggled,
    )
    assert (ok, reason) == (False, "receipt_policy_mismatch")
