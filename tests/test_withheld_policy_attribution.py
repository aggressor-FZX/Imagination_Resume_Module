"""A withheld result must report the policy that actually ran.

The defect this pins: `_mode_from_payload` returned ``(None, None)`` whenever the
request did not *spell* the mode, even though an omitted mode resolves to the
documented default (imaginative / imaginative-v1) and the pipeline runs that
default. So the website — which never sends the field — received a refusal with
no policy attribution, while a caller who sent the field explicitly did. The
difference reflected request spelling, not behaviour.

A withheld result has always carried fixed public metadata (a reason code, a
review message). The effective mode is the same class of fact: the caller chose
it (or accepted the default) and is entitled to know which policy refused their
document. It contains no candidate content.

The negative control matters as much as the fix: a value that cannot be
interpreted must still report nothing rather than assert a policy that may not
have run.
"""

import pytest

import app as imaginator_app


class _Payload:
    """Stand-in for AnalysisRequest: only the one attribute is read."""

    def __init__(self, mode="__absent__"):
        if mode != "__absent__":
            self.skill_inference_mode = mode


def _mode(mode="__absent__"):
    return imaginator_app._mode_from_payload(_Payload(mode))


# ── The fix ───────────────────────────────────────────────────────────────


def test_an_omitted_mode_reports_the_default_policy():
    """The common case: the website sends no field and the default runs."""
    assert _mode() == ("imaginative", "imaginative-v1")


def test_an_explicit_none_reports_the_default_policy():
    """JSON `null` and an absent key mean the same thing to the pipeline."""
    assert _mode(None) == ("imaginative", "imaginative-v1")


def test_an_empty_string_reports_the_default_policy():
    """Both transports send "" for an unset control; it is not a third mode."""
    assert _mode("") == ("imaginative", "imaginative-v1")


def test_whitespace_only_reports_the_default_policy():
    assert _mode("   ") == ("imaginative", "imaginative-v1")


def test_an_explicit_conservative_is_preserved():
    assert _mode("conservative") == ("conservative", "conservative-v1")


def test_an_explicit_imaginative_is_preserved():
    assert _mode("imaginative") == ("imaginative", "imaginative-v1")


@pytest.mark.parametrize("spelling", ["Conservative", " CONSERVATIVE ", "conservative\t"])
def test_the_mode_is_read_case_and_whitespace_insensitively(spelling):
    assert _mode(spelling) == ("conservative", "conservative-v1")


def test_the_enum_is_accepted_directly():
    from skill_inference import SkillInferenceMode

    assert _mode(SkillInferenceMode.CONSERVATIVE) == ("conservative", "conservative-v1")


# ── Negative control: do not guess ────────────────────────────────────────


def test_an_uninterpretable_value_reports_nothing():
    """Asserting a policy that may not have run would be worse than silence."""
    assert _mode("balanced") == (None, None)


def test_a_non_string_value_reports_nothing():
    assert _mode(123) == (None, None)


# ── What the withholding path does with it ────────────────────────────────


def _metrics():
    """A valid RunMetrics payload — the withhold path requires a complete one."""
    return {
        "calls": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "failures": [],
    }


def test_the_withheld_response_carries_the_default_policy():
    """End of the chain: the field the client actually reads."""
    mode, policy = _mode()
    response = imaginator_app._withhold_correction(
        reason="source_facts_missing",
        pipeline_status="completed",
        metrics=_metrics(),
        start_time=0.0,
        skill_inference_mode=mode,
        skill_inference_policy_version=policy,
    )
    assert response.skill_inference_mode == "imaginative"
    assert response.skill_inference_policy_version == "imaginative-v1"
    # Withholding is unchanged: no deliverable content accompanies the metadata.
    assert response.requires_review is True
    assert response.publication_receipt is None
    assert response.approved_export_payload is None
    assert response.final_written_section_markdown is None


def test_withholding_still_says_nothing_when_the_mode_is_uninterpretable():
    mode, policy = _mode("balanced")
    response = imaginator_app._withhold_correction(
        reason="source_facts_missing",
        pipeline_status="completed",
        metrics=_metrics(),
        start_time=0.0,
        skill_inference_mode=mode,
        skill_inference_policy_version=policy,
    )
    assert response.skill_inference_mode is None
    assert response.skill_inference_policy_version is None
