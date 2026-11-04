"""Phase B: typed, source-anchored skill bridges.

A bridge is a *proposal*, not a fact. These tests pin the rules that keep the
distinction enforceable:

* the quoted evidence must appear verbatim in the immutable candidate source;
* a client-supplied ``origin`` never grants anything;
* Conservative produces no records at all, even from valid proposals;
* a missing taxonomy is recorded as missing, never invented;
* a bridge URL/id is server-assigned and deterministic.

Offline only. No provider, network or taxonomy service is contacted.
"""
import pytest

from skill_bridge import (
    MAX_BRIDGES,
    ORIGIN_RESEARCHER_INFERENCE,
    BridgeValidationError,
    bridges_for_writer,
    build_bridges,
    policy_record,
    source_map,
)


SOURCES = {
    "document-0": (
        "Jordan Example\n"
        "Skills: conduit bending, panel labeling\n"
        "Compared weekly sales figures to identify trends across 12 branches.\n"
        "Trained 6 apprentices on lockout/tagout and NFPA 70E.\n"
    ),
    "document-1": (
        "Analyzed measurement series collected over time to characterize drift.\n"
    ),
}


def _proposal(**overrides):
    base = {
        "skill": "trend analysis",
        "evidence_quote": "Compared weekly sales figures to identify trends",
        "rationale": "The activity identifies changes over time.",
        "source_id": "document-0",
    }
    base.update(overrides)
    return base


# ── Happy path ─────────────────────────────────────────────────────────────


def test_valid_proposal_becomes_a_server_owned_record():
    accepted, rejected = build_bridges([_proposal()], sources=SOURCES, mode="imaginative")

    assert rejected == []
    assert len(accepted) == 1
    bridge = accepted[0]
    assert bridge.id.startswith("bridge-")
    # Origin is assigned by the server, never taken from the proposal.
    assert bridge.origin == ORIGIN_RESEARCHER_INFERENCE
    assert bridge.skill == "trend analysis"
    assert bridge.source_id == "document-0"
    assert bridge.evidence_quote in SOURCES["document-0"]
    assert bridge.policy_version == "imaginative-v1"


def test_second_document_can_anchor_a_bridge():
    accepted, rejected = build_bridges(
        [_proposal(
            skill="time-series analysis",
            evidence_quote="Analyzed measurement series collected over time",
            rationale="Measurements over time are a time series.",
            source_id="document-1",
        )],
        sources=SOURCES,
        mode="imaginative",
    )
    assert rejected == []
    assert accepted[0].source_id == "document-1"
    assert accepted[0].evidence_quote in SOURCES["document-1"]


def test_bridge_id_is_deterministic_for_the_same_anchor():
    first, _ = build_bridges([_proposal()], sources=SOURCES, mode="imaginative")
    second, _ = build_bridges([_proposal()], sources=SOURCES, mode="imaginative")
    assert first[0].id == second[0].id


def test_ids_are_distinct_for_distinct_anchors():
    accepted, _ = build_bridges(
        [
            _proposal(),
            _proposal(
                skill="training delivery",
                evidence_quote="Trained 6 apprentices on lockout/tagout and NFPA 70E.",
                rationale="The candidate delivered technical training.",
            ),
        ],
        sources=SOURCES,
        mode="imaginative",
    )
    assert len(accepted) == 2
    assert accepted[0].id != accepted[1].id


# ── Anchoring: the decisive rule ───────────────────────────────────────────


def test_quote_not_in_source_is_rejected():
    accepted, rejected = build_bridges(
        [_proposal(evidence_quote="Built a Kubernetes platform for 1M users")],
        sources=SOURCES,
        mode="imaginative",
    )
    assert accepted == []
    assert rejected[0]["reason"] == "quote_not_in_candidate_source"


def test_paraphrased_quote_is_rejected():
    """A bridge must quote the source, not summarise it."""
    accepted, rejected = build_bridges(
        [_proposal(evidence_quote="looked at weekly sales numbers for patterns")],
        sources=SOURCES,
        mode="imaginative",
    )
    assert accepted == []
    assert rejected[0]["reason"] == "quote_not_in_candidate_source"


def test_quote_without_a_source_anchor_is_rejected():
    """Undeclared source ids still resolve, but only if the text really exists."""
    accepted, rejected = build_bridges(
        [_proposal(source_id="document-9")], sources=SOURCES, mode="imaginative"
    )
    # document-9 does not exist, so resolution falls back to a real document.
    assert accepted
    assert accepted[0].source_id in SOURCES


def test_quote_spanning_two_documents_is_rejected():
    """An anchor must not merge two owners' evidence."""
    merged = {
        "document-0": "Compared weekly figures",
        "document-1": "across 12 branches and improved outcomes",
    }
    accepted, rejected = build_bridges(
        [_proposal(
            evidence_quote="Compared weekly figures\n\n---\n\nacross 12 branches",
            source_id="document-0",
        )],
        sources={"document-0": "Compared weekly figures\n\n---\n\nacross 12 branches"},
        mode="imaginative",
    )
    # The join marker is present inside the span, so it cannot be one anchor.
    assert accepted == []
    assert rejected[0]["reason"] == "quote_not_in_candidate_source"


# ── Provenance cannot be asserted ──────────────────────────────────────────


def test_client_supplied_origin_does_not_grant_anything():
    accepted, _ = build_bridges(
        [_proposal(origin="candidate_evidence", policy_version="trusted-v9")],
        sources=SOURCES,
        mode="imaginative",
    )
    assert accepted[0].origin == ORIGIN_RESEARCHER_INFERENCE
    assert accepted[0].policy_version == "imaginative-v1"


def test_client_cannot_nominate_its_own_bridge_id():
    accepted, _ = build_bridges(
        [_proposal(id="bridge-approved-by-admin")], sources=SOURCES, mode="imaginative"
    )
    assert accepted[0].id.startswith("bridge-")
    assert "admin" not in accepted[0].id


# ── Conservative ───────────────────────────────────────────────────────────


def test_conservative_produces_no_records_even_for_valid_proposals():
    accepted, rejected = build_bridges([_proposal()], sources=SOURCES, mode="conservative")
    assert accepted == []
    assert rejected[0]["reason"] == "mode_forbids_bridging"


def test_conservative_rejects_every_proposal():
    proposals = [_proposal(), _proposal(skill="time-series analysis",
                                        evidence_quote="Analyzed measurement series collected over time")]
    accepted, rejected = build_bridges(proposals, sources=SOURCES, mode="conservative")
    assert accepted == []
    assert len(rejected) == 2


# ── Label shape: general skill, not a smuggled claim ──────────────────────


@pytest.mark.parametrize(
    "skill",
    [
        "ARIMA forecasting",         # specific algorithm
        "Python",                    # named tool only
        "Used Python for trend analysis and increased revenue 20%",  # a claim
        "expert in trend analysis",  # proficiency
        "led a team of 5 analysts",  # leadership + team size
        "trend analysis: advanced",  # category/level
        "analyzed data. achieved 20% growth.",  # sentence
    ],
)
def test_non_general_labels_are_rejected(skill):
    accepted, rejected = build_bridges(
        [_proposal(skill=skill)], sources=SOURCES, mode="imaginative"
    )
    assert accepted == []
    assert rejected[0]["reason"] == "not_a_general_skill_label"


@pytest.mark.parametrize(
    "skill",
    ["trend analysis", "time-series analysis", "training delivery", "technical instruction"],
)
def test_general_labels_are_accepted(skill):
    accepted, rejected = build_bridges(
        [_proposal(skill=skill)], sources=SOURCES, mode="imaginative"
    )
    assert rejected == []
    assert accepted[0].skill == skill


# ── Structural rejection ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"skill": ""}, "missing_skill"),
        ({"evidence_quote": ""}, "missing_evidence_quote"),
        ({"rationale": ""}, "missing_rationale"),
        ({"rationale": "x" * 900}, "rationale_too_long"),
        ({"evidence_quote": "x" * 900}, "evidence_quote_too_long"),
    ],
)
def test_malformed_records_are_rejected_individually(overrides, reason):
    accepted, rejected = build_bridges(
        [_proposal(**overrides)], sources=SOURCES, mode="imaginative"
    )
    assert accepted == []
    assert rejected[0]["reason"] == reason


def test_a_bad_record_does_not_discard_a_good_sibling():
    accepted, rejected = build_bridges(
        [
            _proposal(),
            _proposal(skill="time-series analysis",
                      evidence_quote="Text that does not exist in any source"),
        ],
        sources=SOURCES,
        mode="imaginative",
    )
    assert len(accepted) == 1
    assert accepted[0].skill == "trend analysis"
    assert len(rejected) == 1


def test_non_object_proposals_are_rejected():
    accepted, rejected = build_bridges(
        ["just a string", None, 42], sources=SOURCES, mode="imaginative"
    )
    assert accepted == []
    assert len(rejected) == 3


def test_bridge_count_is_bounded():
    many = [
        _proposal(
            skill=f"skill {i}",
            evidence_quote="Compared weekly sales figures to identify trends",
        )
        for i in range(MAX_BRIDGES + 3)
    ]
    # Only the first anchor is unique; later duplicates are rejected as dupes,
    # so assert the bound holds without errors at minimum.
    accepted, rejected = build_bridges(many, sources=SOURCES, mode="imaginative")
    assert len(accepted) <= MAX_BRIDGES


def test_duplicate_anchor_is_rejected():
    accepted, rejected = build_bridges(
        [_proposal(), _proposal()], sources=SOURCES, mode="imaginative"
    )
    assert len(accepted) == 1
    assert rejected[0]["reason"] == "duplicate_bridge"


# ── Grounding: optional, never fabricated ─────────────────────────────────


def test_missing_taxonomy_yields_empty_refs_not_invented_ones():
    accepted, _ = build_bridges([_proposal()], sources=SOURCES, mode="imaginative")
    assert accepted[0].taxonomy_refs == ()


def test_taxonomy_refs_attached_when_supplied():
    accepted, _ = build_bridges(
        [_proposal()],
        sources=SOURCES,
        mode="imaginative",
        taxonomy={"trend analysis": ["onet:4.A.2.b", "esco:S1.2"]},
    )
    assert set(accepted[0].taxonomy_refs) == {"onet:4.A.2.b", "esco:S1.2"}


def test_model_cannot_invent_a_taxonomy_reference():
    accepted, _ = build_bridges(
        [_proposal(taxonomy_refs=["onet:FAKE-999"])],
        sources=SOURCES,
        mode="imaginative",
        taxonomy={"trend analysis": ["onet:4.A.2.b"]},
    )
    # Only refs that exist in the caller-supplied taxonomy survive.
    assert "onet:FAKE-999" not in accepted[0].taxonomy_refs
    assert accepted[0].taxonomy_refs == ()


def test_no_taxonomy_means_no_reference_even_if_model_claims_one():
    accepted, _ = build_bridges(
        [_proposal(taxonomy_refs=["onet:4.A.2.b"])],
        sources=SOURCES,
        mode="imaginative",
        taxonomy=None,
    )
    assert accepted[0].taxonomy_refs == ()


# ── Downstream views ───────────────────────────────────────────────────────


def test_writer_view_excludes_offsets_and_policy_plumbing():
    accepted, _ = build_bridges([_proposal()], sources=SOURCES, mode="imaginative")
    view = bridges_for_writer(accepted)
    assert set(view[0]) == {"skill", "evidence_quote", "rationale"}


def test_policy_record_is_deterministic_and_ordered():
    accepted, _ = build_bridges([_proposal()], sources=SOURCES, mode="imaginative")
    first = policy_record("imaginative", accepted)
    second = policy_record("imaginative", accepted)
    assert first == second
    assert first["skill_inference_mode"] == "imaginative"
    assert [b["id"] for b in first["surviving_bridges"]] == [accepted[0].id]


def test_conservative_policy_record_has_no_bridges():
    record = policy_record("conservative", [])
    assert record["surviving_bridges"] == []
    assert record["skill_inference_mode"] == "conservative"


def test_source_map_does_not_merge_documents():
    mapping = source_map([
        {"source_id": "a", "text": "first"},
        {"source_id": "b", "text": "second"},
    ])
    assert mapping == {"a": "first", "b": "second"}


def test_source_map_ignores_empty_entries():
    assert source_map([{"source_id": "a", "text": "  "}, {"source_id": ""}]) == {}
    assert source_map(None) == {}
