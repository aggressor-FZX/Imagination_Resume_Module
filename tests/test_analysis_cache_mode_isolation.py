"""Phase E: cache identity and cross-mode isolation.

The defect this closes: the analysis cache key was built from the candidate text,
job ad, structured inputs and confidence threshold — but **not** the skill-inference
mode. Two requests with identical text but different modes hashed the same, so a
Conservative request could be served an Imaginative result whose bridge labels the
caller never asked for. That is a permission leak across requests, not a cache miss.

Rules pinned here:

* mode and policy version are part of the cache identity;
* an omitted mode hashes as the default (imaginative) — the documented default;
* an unknown mode does **not** collapse onto the default — it gets its own
  identity, so it cannot inherit a legitimate result;
* cached values are returned only for the identity they were stored under.

Offline only. The cache is the real module-level cache; it is cleared per test.
"""
import pytest

import imaginator_flow as flow

from skill_inference import SkillInferenceMode, policy_version_for


RESUME = "Python developer with five years of experience building pipelines."
JOB_AD = "Senior Data Analyst. Requires SQL, Python and trend analysis."


@pytest.fixture(autouse=True)
def clear_cache():
    flow.ANALYSIS_CACHE.clear()
    yield
    flow.ANALYSIS_CACHE.clear()


def _key(mode=None, policy_version=None, **overrides):
    kwargs = dict(
        resume_text=RESUME,
        job_ad=JOB_AD,
        extracted_skills_json={},
        domain_insights_json={},
        confidence_threshold=0.7,
    )
    kwargs.update(overrides)
    return flow._make_analysis_cache_key(
        skill_inference_mode=mode,
        skill_inference_policy_version=policy_version,
        **kwargs,
    )


# ── identity ───────────────────────────────────────────────────────────────


def test_the_two_modes_get_different_cache_identities():
    assert _key("imaginative") != _key("conservative")


def test_policy_version_differences_change_the_identity():
    """A policy bump is a behaviour change, so it must be a different entry."""
    assert _key("imaginative", "imaginative-v1") != _key(
        "imaginative", "imaginative-v2"
    )


def test_an_omitted_mode_hashes_as_the_documented_default():
    """Omission means imaginative — the same request, not a new one."""
    assert _key(None) == _key("imaginative")


@pytest.mark.parametrize("absent", ["", "   "])
def test_blank_mode_hashes_as_the_default(absent):
    assert _key(absent) == _key("imaginative")


def test_an_unknown_mode_does_not_collapse_onto_the_default():
    """Otherwise a bogus value would inherit a legitimate cached result."""
    unknown = _key("balanced")
    assert unknown != _key("imaginative")
    assert unknown != _key("conservative")


def test_existing_identity_components_still_separate_requests():
    """The pre-existing components must keep working."""
    assert _key("imaginative") != _key("imaginative", resume_text="Different text")
    assert _key("imaginative") != _key("imaginative", job_ad="Different ad")
    assert _key("imaginative") != _key("imaginative", confidence_threshold=0.9)


def test_identity_is_deterministic():
    assert _key("conservative") == _key("conservative")


# ── round trip through the real cache ─────────────────────────────────────


def test_a_cached_value_is_only_returned_for_its_own_mode():
    imaginative_key = _key("imaginative")
    conservative_key = _key("conservative")
    imaginative_result = {"verdict": "imaginative-result",
                          "skill_bridges": [{"id": "b1", "skill": "trend analysis"}]}
    flow._analysis_cache_set(imaginative_key, imaginative_result)

    # The mode that stored it gets it back.
    assert flow._analysis_cache_get(imaginative_key) == imaginative_result
    # The other mode does not.
    assert flow._analysis_cache_get(conservative_key) is None


def test_a_conservative_result_never_leaks_to_an_imaginative_request():
    conservative_key = _key("conservative")
    flow._analysis_cache_set(conservative_key, {"verdict": "conservative-result"})
    assert flow._analysis_cache_get(_key("imaginative")) is None


def test_both_modes_can_coexist():
    flow._analysis_cache_set(_key("imaginative"), {"mode": "imaginative"})
    flow._analysis_cache_set(_key("conservative"), {"mode": "conservative"})
    assert flow._analysis_cache_get(_key("imaginative"))["mode"] == "imaginative"
    assert flow._analysis_cache_get(_key("conservative"))["mode"] == "conservative"


def test_an_expired_entry_is_not_returned(monkeypatch):
    key = _key("imaginative")
    flow._analysis_cache_set(key, {"verdict": "stale"})
    # Force expiry without sleeping.
    expires_at, value = flow.ANALYSIS_CACHE[key]
    flow.ANALYSIS_CACHE[key] = (-1.0, value)
    assert flow._analysis_cache_get(key) is None


# ── the mode contract is the single source of truth ───────────────────────


def test_cache_key_uses_the_shared_normalisation_point():
    """An alias spelling must land on the same identity as the canonical value."""
    assert _key("IMAGINATIVE") == _key("imaginative")
    assert _key("  Conservative  ") == _key("conservative")


def test_policy_version_defaults_from_the_mode():
    """When the caller does not pass a version, the contract supplies it."""
    imaginative = _key("imaginative")
    explicit = _key(
        "imaginative", policy_version_for(SkillInferenceMode.IMAGINATIVE)
    )
    assert imaginative == explicit
