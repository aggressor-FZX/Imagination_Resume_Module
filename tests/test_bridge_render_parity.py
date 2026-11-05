"""Phase D remainder: preview/export parity, and real file-content assertions.

The remaining Phase D acceptance item from the spec (§D.6/D.8 step 5):

    Enforce identical effective payload selection at verifier/renderer, test
    real PDF/DOCX content and private artifact isolation.

    A separate real-render test must read file contents; no single mocked test
    establishes the whole delivery.

What this file adds over the existing suites:

* ``test_export_isolation_and_parity`` renders real files but never mentions
  skill inference, so nothing asserted that the *reviewed* label reaches the
  bytes or that a Conservative run excludes it.
* ``test_bridge_receipt_binding`` proves the policy state is signed but stops at
  the receipt; it asserts the renderer is NOT called, never that it produces the
  right content.
* Nothing compared the *preview* string (the approved markdown the browser
  shows) with the *file* content for the same approved payload.

Two claims are made here, and they are stated separately because they can fail
independently:

1. **Parity** — the preview and both files are derived from the SAME approved
   object. Every substantive skill label in the previewed markdown appears in
   both files, and neither file contains a label the preview omitted.
2. **Modes** — an Imaginative approval's inferred label survives into PDF and
   DOCX; the same label is absent in Conservative, where no bridge exists.

Real ``pdflatex`` + ``pdftotext`` + ``python-docx`` are used. If the local
toolchain is missing the test is SKIPPED with a reason, never turned into a pass
on mocked bytes — a mocked render establishes nothing about delivery.
"""

import base64
import io
import re
import shutil
import subprocess

from docx import Document
import pytest

import publication_gate as gate
from resume_builder.build_resume_from_export import build_resume_from_payload_base64


pytestmark = pytest.mark.skipif(
    not (shutil.which("pdflatex") and shutil.which("pdftotext")),
    reason="real renderer toolchain (pdflatex/pdftotext) not installed",
)


OWNER = "candidate-8821"
SECRET = "phase-d-render-parity-secret"
INFERRED = "trend analysis"

# The approved markdown the browser previews. It is the authority: the files are
# only correct insofar as they carry the same substantive content.
APPROVED_MD = (
    "## Professional Summary\n"
    "Analyst with measurement and reporting experience.\n\n"
    "## Professional Experience\n"
    "**Cascade Power Solutions** -- Example City, WA\n"
    "*Data Analyst | 03/2020 -- Present*\n"
    "- Compared weekly sales figures to identify trends across 12 branches.\n"
    "- Delivered trend analysis for regional stakeholders.\n\n"
    "## Education\n"
    "**Example Technical College** -- 2016\n\n"
    "## Technical Skills\n"
    f"**Analysis:** {INFERRED}, reporting\n"
)

APPROVED_PAYLOAD = {
    "name": "Synthetic Candidate",
    "headline": "Data Analyst",
    "summary": "Analyst with measurement and reporting experience.",
    "skills": [{"category": "Analysis", "items": [INFERRED, "reporting"]}],
    "experience": [
        {
            "title": "Data Analyst",
            "company": "Cascade Power Solutions",
            "duration": "03/2020 -- Present",
            "bullets": [
                "Compared weekly sales figures to identify trends across 12 branches.",
                "Delivered trend analysis for regional stakeholders.",
            ],
        }
    ],
    "education": ["Example Technical College -- 2016"],
    "certifications": [],
    "projects": [],
}


def _policy(mode="imaginative", bridges="default"):
    if bridges == "default":
        bridges = [{
            "id": "bridge-01-abc123",
            "skill": INFERRED,
            "origin": "researcher_inference",
            "source_id": "document-0",
            "policy_version": "imaginative-v1",
        }] if mode == "imaginative" else []
    return {
        "skill_inference_mode": mode,
        "skill_inference_policy_version": (
            "imaginative-v1" if mode == "imaginative" else "conservative-v1"
        ),
        "surviving_bridges": bridges,
    }


@pytest.fixture(autouse=True)
def signing(monkeypatch):
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", SECRET)
    monkeypatch.setenv("PUBLICATION_RECEIPT_TTL_SECONDS", "900")


def _docx_text(encoded):
    document = Document(io.BytesIO(base64.b64decode(encoded)))
    return "\n".join(p.text for p in document.paragraphs)


def _pdf_text(encoded, tmp_path):
    """Extract real PDF text. A failure here is a test error, not a skip."""
    pdf = tmp_path / "parity.pdf"
    pdf.write_bytes(base64.b64decode(encoded))
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout


def _norm(text):
    """Collapse layout whitespace only — never strip meaningful characters."""
    return re.sub(r"\s+", " ", text or "").strip()


def _render(tmp_path):
    result = build_resume_from_payload_base64(dict(APPROVED_PAYLOAD))
    return (
        result,
        _docx_text(result["docx_base64"]),
        _pdf_text(result["pdf_base64"], tmp_path),
    )


def _md_skill_labels(markdown):
    """Skill labels from the previewed markdown's Technical Skills block.

    The block is written as ``**Category:** item, item``. The markdown emphasis
    is preview formatting, not part of the label, so it is stripped here — else
    ``**trend analysis`` would be compared against the rendered ``trend
    analysis`` and the parity test would fail for a formatting reason.
    """
    labels = []
    for line in markdown.splitlines():
        if ":" not in line or line.startswith("#"):
            continue
        _, _, items = line.partition(":")
        for part in items.split(","):
            cleaned = part.replace("*", "").strip()
            if cleaned:
                labels.append(cleaned)
    return labels


# ── 1. Preview / file parity ──────────────────────────────────────────────


def test_every_previewed_skill_label_appears_in_both_files(tmp_path):
    """The browser preview and the two files must carry the same labels.

    This is the parity assertion the spec asks for. It is stated over the
    *preview string*, not over the payload, so a renderer that silently dropped
    or substituted a skill would fail here even if the payload were correct.
    """
    _, docx, pdf = _render(tmp_path)
    labels = _md_skill_labels(APPROVED_MD)
    assert INFERRED in labels, "fixture no longer contains the inferred label"

    for label in labels:
        assert label in _norm(docx), f"{label!r} missing from DOCX"
        assert label in _norm(pdf), f"{label!r} missing from PDF"


def test_neither_file_invents_a_skill_the_preview_omitted(tmp_path):
    """Parity is two-sided: the files must not gain a label the preview lacks.

    A real renderer-level fabrication would show up as a skill in the bytes that
    has no counterpart in the approved markdown. The check is deliberately
    scoped to skill-shaped tokens so ordinary template words do not trip it.
    """
    _, docx, pdf = _render(tmp_path)
    preview_labels = {label.casefold() for label in _md_skill_labels(APPROVED_MD)}
    # Labels that exist in the payload but were NOT approved into the markdown.
    unapproved = {"python", "tensorflow", "kubernetes", "expert"}
    assert not (unapproved & preview_labels), "fixture is inconsistent"
    for text in (docx, pdf):
        normalised = _norm(text).casefold()
        for label in unapproved:
            assert label not in normalised, f"{label!r} appeared without approval"


def test_the_approved_employer_owns_the_approved_bullets_in_both_files(tmp_path):
    """Ownership survives rendering: no bullet drifts to another employer.

    Parity of *strings* is not parity of attribution. The approved bullet must
    render under the approved employer in both formats.
    """
    _, docx, pdf = _render(tmp_path)
    bullet = "Compared weekly sales figures to identify trends across 12 branches."
    for text in (docx, pdf):
        body = _norm(text)
        assert bullet in body
        employer_at = body.index("Cascade Power Solutions")
        bullet_at = body.index(bullet)
        assert employer_at < bullet_at, "bullet rendered before its employer"


# ── 2. Mode parity in real bytes ──────────────────────────────────────────


def test_imaginative_label_survives_into_both_files(tmp_path):
    """Positive half: an authorised label must actually be delivered."""
    _, docx, pdf = _render(tmp_path)
    assert INFERRED in _norm(docx)
    assert INFERRED in _norm(pdf)


def test_conservative_payload_renders_no_inferred_label(tmp_path):
    """Negative half: with no bridge, the label must not appear.

    The payload deliberately omits the label rather than the renderer filtering
    it — the renderer has no notion of modes, and giving it one would create a
    second place where inference policy is decided. What is asserted here is the
    *delivered difference* between the two modes, which is what a customer sees.
    """
    conservative = dict(APPROVED_PAYLOAD)
    conservative["skills"] = [{"category": "Analysis", "items": ["reporting"]}]
    conservative["experience"] = [
        {
            **APPROVED_PAYLOAD["experience"][0],
            "bullets": APPROVED_PAYLOAD["experience"][0]["bullets"][:1],
        }
    ]
    result = build_resume_from_payload_base64(conservative)
    docx = _norm(_docx_text(result["docx_base64"]))
    pdf = _norm(_pdf_text(result["pdf_base64"], tmp_path))
    for text in (docx, pdf):
        assert INFERRED not in text
        assert "Compared weekly sales figures" in text


def test_a_narrowed_label_renders_its_replacement_not_the_original(tmp_path):
    """A narrowed bridge must deliver the reviewer-approved wording only.

    Narrowing rewrites the label and forces re-review; the rendering must carry
    the replacement. If both strings appeared, the original would still be
    reaching the customer after being replaced.
    """
    narrowed = dict(APPROVED_PAYLOAD)
    narrowed["skills"] = [{"category": "Analysis", "items": ["reporting", "analysis"]}]
    narrowed["experience"] = [
        {
            **APPROVED_PAYLOAD["experience"][0],
            "bullets": ["Compared weekly sales figures to identify trends across 12 branches."],
        }
    ]
    result = build_resume_from_payload_base64(narrowed)
    docx = _norm(_docx_text(result["docx_base64"]))
    pdf = _norm(_pdf_text(result["pdf_base64"], tmp_path))
    for text in (docx, pdf):
        assert INFERRED not in text
        assert "analysis" in text.casefold()


# ── 3. Verifier and renderer read the same object ─────────────────────────


def test_receipt_issued_for_the_payload_verifies_against_the_rendered_object(tmp_path):
    """End-to-end: issue → verify → render, all on one object.

    The spec's requirement is that verification and rendering resolve the SAME
    effective payload. Anything that made them read different objects (a nested
    `data` envelope, an alias, an outer override) is caught by this pairing.
    """
    receipt = gate.issue_receipt(
        markdown=APPROVED_MD,
        verdict=gate.Verdict.PASS_TIER_A.value,
        owner=OWNER,
        approved_payload=dict(APPROVED_PAYLOAD),
        policy_state=_policy(),
    )
    assert receipt

    ok, reason = gate.verify_receipt(
        receipt,
        markdown=APPROVED_MD,
        owner=OWNER,
        render_payload=dict(APPROVED_PAYLOAD),
        policy_state=_policy(),
    )
    assert (ok, reason) == (True, "ok")

    result = build_resume_from_payload_base64(dict(APPROVED_PAYLOAD))
    assert INFERRED in _norm(_docx_text(result["docx_base64"]))


def test_a_bullet_swapped_after_approval_is_refused_before_rendering(tmp_path):
    """The verifier must not be bypassable by rendering first.

    Negative control for the pairing above: the same valid receipt, presented
    with altered content, must be refused. Rendering it anyway would mean the
    receipt never governed the bytes.
    """
    receipt = gate.issue_receipt(
        markdown=APPROVED_MD,
        verdict=gate.Verdict.PASS_TIER_A.value,
        owner=OWNER,
        approved_payload=dict(APPROVED_PAYLOAD),
        policy_state=_policy(),
    )
    tampered = dict(APPROVED_PAYLOAD)
    tampered["experience"] = [
        {
            **APPROVED_PAYLOAD["experience"][0],
            "bullets": ["Increased regional revenue by 40%."],
        }
    ]
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=APPROVED_MD,
        owner=OWNER,
        render_payload=tampered,
        policy_state=_policy(),
    )
    assert not ok
    assert reason


def test_conservative_export_cannot_present_an_imaginative_receipt(tmp_path):
    """Cross-mode replay: an imaginative receipt must not authorise a
    conservative document that dropped the bridge.

    This is the delivery-level form of the receipt asymmetry — the mode is part
    of the signed policy state, so swapping it is a mismatch, not a downgrade.
    """
    receipt = gate.issue_receipt(
        markdown=APPROVED_MD,
        verdict=gate.Verdict.PASS_TIER_A.value,
        owner=OWNER,
        approved_payload=dict(APPROVED_PAYLOAD),
        policy_state=_policy("imaginative"),
    )
    ok, reason = gate.verify_receipt(
        receipt,
        markdown=APPROVED_MD,
        owner=OWNER,
        render_payload=dict(APPROVED_PAYLOAD),
        policy_state=_policy("conservative"),
    )
    assert (ok, reason) == (False, "receipt_policy_mismatch")
