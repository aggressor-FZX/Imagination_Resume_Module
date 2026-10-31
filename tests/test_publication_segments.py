"""Synthetic adversarial fixtures for the security-sensitive Markdown segmenter."""
import copy

import pytest

import publication_gate as gate


ACHIEVEMENT = "Reduced diagnostic time by 30 percent across the service region"
MAINTENANCE = "Maintained electrical equipment throughout the service region"


def document(first="**Employer Alpha**", second="**Employer Beta**", bullet=None):
    return (
        "## Professional Experience\n" + first + "\n- "
        + (bullet or ACHIEVEMENT) + "\n" + second + "\n- "
        + MAINTENANCE + "\n## Education\n**Synthetic College**\n"
        "- Diploma in electrical engineering\n"
    )


def payload():
    return {"experience": [
        {"company": "Employer Alpha", "bullets": [ACHIEVEMENT]},
        {"company": "Employer Beta", "bullets": [MAINTENANCE]},
    ], "education": ["Diploma in electrical engineering, Synthetic College"]}


@pytest.mark.parametrize("headings", [
    ("### Employer Alpha", "### Employer Beta"),
    ("**Employer Alpha**", "**Employer Beta**"),
    ("### **Employer Alpha**", "### **Employer Beta**"),
])
def test_recognized_odd_headings_scope_employers_separately(headings):
    md = document(*headings)
    sections = gate.approved_segments(md)
    career = [s for s in sections if s["section"] == "experience" and s["is_block"]]
    assert len(career) == 2
    assert not gate._block_binding_violations(payload(), md, sections)
    changed = payload()
    changed["experience"][1]["bullets"].append(ACHIEVEMENT)
    assert "career_block_reassigned" in {
        v["reason"] for v in gate._block_binding_violations(changed, md, sections)
    }


def test_unknown_plain_headings_fail_loudly_not_one_global_scope():
    md = document("Employer Alpha | Technician", "Employer Beta | Technician")
    assert "career_segmentation_ambiguous" in {
        v["reason"] for v in gate.render_content_violations(payload(), md)
    }


def test_bold_text_inside_nested_bullets_is_not_an_employer():
    md = document(bullet="**Reduced diagnostic time** by 30 percent\n"
                  "  - **Maintained instruments** during daily inspection")
    segments = gate.approved_segments(md)
    career = [s for s in segments if s["section"] == "experience" and s["is_block"]]
    assert len(career) == 2
    assert len(career[0]["bullets"]) == 2
    assert "Maintained instruments" in career[0]["bullets"][1]


def test_wrapped_bullet_keeps_one_claim_and_its_employer():
    md = document(bullet="Reduced diagnostic time by 30 percent\n"
                  "  across the service region")
    segments = gate.approved_segments(md)
    alpha = next(s for s in segments if s["heading"] == "Employer Alpha")
    assert gate._squash(ACHIEVEMENT) in alpha["claims"]
    assert len(alpha["bullets"]) == 1
    assert not gate._preservation_violations(payload(), segments)


@pytest.mark.parametrize("label", ["## Education", "**Education**", "Education"])
def test_section_labels_never_become_employers(label):
    md = document().replace("## Education", label)
    segments = gate.approved_segments(md)
    education = next(s for s in segments if s["heading"] == "Education")
    assert not education["is_block"]
    assert education["section"] == "other"


def test_removed_bullet_in_rendered_section_rejects():
    changed = payload()
    changed["experience"][0]["bullets"] = []
    violations = gate._preservation_violations(changed, gate.approved_segments(document()))
    assert violations and violations[0]["reason"] == "approved_achievement_omitted"


def test_deleted_whole_employer_is_not_a_preservation_exemption():
    changed = payload()
    changed["experience"].pop(0)
    assert gate._preservation_violations(changed, gate.approved_segments(document()))


def test_flat_education_bullet_not_rendered_as_career_is_exempt():
    # Education itself is emitted as a flat entry; it is only the career-bullet
    # preservation rule that must not mistake its degree for employment prose.
    assert not gate._preservation_violations(payload(), gate.approved_segments(document()))


def test_known_limit_reordering_inside_one_approved_block():
    md = document().replace("- " + ACHIEVEMENT, "- " + ACHIEVEMENT + "\n- " + MAINTENANCE)
    changed = copy.deepcopy(payload())
    changed["experience"][0]["bullets"] = [MAINTENANCE, ACHIEVEMENT]
    assert not gate._block_binding_violations(changed, md, gate.approved_segments(md))


def test_known_limit_same_bullet_approved_under_two_employers_is_ambiguous():
    md = document().replace("- " + MAINTENANCE, "- " + ACHIEVEMENT)
    changed = payload()
    changed["experience"][1]["bullets"] = [ACHIEVEMENT]
    # Deliberately characterises a limit, not evidence of general attribution.
    # A duplicate claim has two scopes, so the current checker cannot decide
    # which original employer a subsequent instance belongs to.
    assert not gate._block_binding_violations(changed, md, gate.approved_segments(md))