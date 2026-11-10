"""Candidate identity: supplied, absent, conflicting, tampered.

The rule these pin
------------------
Identity is **caller-supplied and signed**, never inferred from document text:

    payload.candidate_identity  ->  approved_export_payload  ->  receipt
                                (no text parsing at any step)

Why that rule, and not "find the name-like line"
------------------------------------------------
A first line of extracted PDF text can be a job title (`Project Manager`), an
agency, or a recruiter's address. Measured here: `Project Manager` followed by
`recruiter@example.test` is a perfectly plausible candidate header by any
shape-based rule, and binding it prints the wrong person's identity on the
deliverable. So the caller — which knows which upload is the candidate's primary
resume — designates identity explicitly, or the header stays empty.

Every test renders real PDF and DOCX and reads the text back, because the defect
these guard against is a wrong name in a deliverable, not a wrong dict key.
"""
import base64
import copy
import io
import shutil
import subprocess

import pytest
from docx import Document

import app
from resume_builder.build_resume_from_export import parse_resume_payload
from tests.test_json_app_identity_bridge_regressions import analyze_identity
from tests.test_json_bridge_app import client, export  # noqa: F401 - fixture


# All five core sections are present and the document clears the publication
# gate's length floor, because a real deliverable does both.
PRIMARY = (
    '## Professional Summary\n'
    'Operations analyst with nine years of experience in freight reconciliation, '
    'depot inventory control and regional cost reporting. Reconciled high-volume '
    'fuel and parts invoices against dispensing and receiving records, resolved '
    'supplier discrepancies with procurement staff, and prepared the monthly cost '
    'summary the regional manager used to set depot operating targets.\n\n'
    '## Professional Experience\n'
    '**Harborline Transit Authority** -- Tacoma, WA\n'
    '- Reconciled 1,180 fuel invoices against depot dispensing records each '
    'quarter, resolving discrepancies with procurement before month end.\n'
    '- Maintained reorder points for maintenance parts across three depots, '
    'reviewing consumption trends and adjusting stock levels with operations '
    'supervisors during weekly planning meetings.\n\n'
    '## Projects\n\n'
    '## Education\n'
    '**Pierce College** -- Lakewood, WA\n'
    '- Associate of Applied Science in Logistics and Supply Chain Management.\n\n'
    '## Certifications\n'
    '**Certified Logistics Associate**\n\n'
    '## Technical Skills\n'
    '- reconciliation, inventory control, cost reporting, supplier coordination\n')
SUPPLEMENT = (
    '## Professional Summary\n'
    'Additional employment records covering cost analysis and regional reporting '
    'support for freight operations.\n\n'
    '## Professional Experience\n'
    '**Cascade Freight Partners** -- Fife, WA\n'
    '- Prepared the monthly cost summary that the regional manager reviewed, '
    'consolidating fuel, labor and maintenance line items into a single report.\n'
    '- Coordinated with dispatch to verify mileage and fuel surcharge figures '
    'before the summary was circulated to depot leadership.\n\n'
    '## Projects\n\n## Education\n\n## Certifications\n\n'
    '## Technical Skills\n- cost reporting\n')

# The writer's own draft: a faithful rewrite of PRIMARY, not the source text.
# Passing the source as the draft is correctly treated as a fallback by the route.
WRITTEN = PRIMARY.replace(
    'Operations analyst with nine years of experience in freight reconciliation, ',
    'Operations analyst with nine years of experience reconciling freight, '
).replace(
    'Reconciled 1,180 fuel invoices against depot dispensing records each '
    'quarter, resolving discrepancies with procurement before month end.',
    'Reconciled 1,180 fuel invoices against depot dispensing records each '
    'quarter and resolved supplier discrepancies with procurement before month '
    'end.')
assert WRITTEN != PRIMARY

IDENTITY = {'name': 'Jeff Calderon', 'email': 'jeff.calderon@example.test',
            'phone': '253-555-0147', 'location': 'Tacoma, WA',
            'linkedin': 'linkedin.com/in/jeff-calderon',
            'github': 'github.com/jeff-calderon'}


def rendered_texts(client, result, tmp_path):
    response = export(client, result)
    assert response.status_code == 200, response.text
    payload = response.json()
    docx = Document(io.BytesIO(base64.b64decode(payload['docx_base64'])))
    docx_text = '\n'.join(p.text for p in docx.paragraphs)
    pdf_path = tmp_path / 'resume.pdf'
    pdf_path.write_bytes(base64.b64decode(payload['pdf_base64']))
    pdf_text = subprocess.run(['pdftotext', str(pdf_path), '-'], check=True,
                              capture_output=True, text=True).stdout
    return docx_text, pdf_text


requires_pdf = pytest.mark.skipif(
    not shutil.which('pdflatex') or not shutil.which('pdftotext'),
    reason='PDF tools unavailable')


# ── 1. Preservation: the supplied identity reaches both deliverables ───────

@requires_pdf
def test_supplied_identity_is_signed_and_rendered_in_both_formats(client, monkeypatch, tmp_path):
    result, _ = analyze_identity(
        client, monkeypatch, PRIMARY, WRITTEN,
        documents=[('a', PRIMARY)], extra={'candidate_identity': IDENTITY})

    assert result['processing_status'] == 'completed', result
    payload = result['approved_export_payload']
    parsed = parse_resume_payload(payload)
    assert parsed.name == IDENTITY['name']
    assert parsed.email == IDENTITY['email']
    assert parsed.phone == IDENTITY['phone']
    assert parsed.location == IDENTITY['location']

    docx_text, pdf_text = rendered_texts(client, result, tmp_path)
    for text in (docx_text, pdf_text):
        assert 'Jeff' in text and 'Calderon' in text, text[:400]
        assert 'jeff.calderon@example.test' in text
        assert '253-555-0147' in text
        # Content is untouched by identity binding.
        assert 'Harborline Transit Authority' in text
        assert '1,180' in text


@requires_pdf
def test_identity_binding_does_not_change_the_approved_markdown(client, monkeypatch, tmp_path):
    """Identity is an export-header concern; it must not rewrite the document."""
    plain, _ = analyze_identity(client, monkeypatch, PRIMARY, WRITTEN,
                                documents=[('a', PRIMARY)])
    with_identity, _ = analyze_identity(
        client, monkeypatch, PRIMARY, WRITTEN, documents=[('a', PRIMARY)],
        extra={'candidate_identity': IDENTITY})
    assert with_identity['final_written_section_markdown'] == \
        plain['final_written_section_markdown']


# ── 2. Absent identity: empty header, never a guess ────────────────────────

@requires_pdf
def test_absent_identity_leaves_the_header_empty(client, monkeypatch, tmp_path):
    result, _ = analyze_identity(client, monkeypatch, PRIMARY, WRITTEN,
                                 documents=[('a', PRIMARY)])
    assert result['processing_status'] == 'completed', result
    parsed = parse_resume_payload(result['approved_export_payload'])
    assert not parsed.name and not parsed.email and not parsed.phone

    docx_text, pdf_text = rendered_texts(client, result, tmp_path)
    for text in (docx_text, pdf_text):
        # No name-shaped guess appears, and the content still renders.
        assert 'Harborline Transit Authority' in text
        assert 'Jeff' not in text and 'Calderon' not in text


def test_identity_forged_on_the_request_body_is_ignored(client, monkeypatch):
    """Fields the model never validates cannot smuggle an unsigned name."""
    result, _ = analyze_identity(
        client, monkeypatch, PRIMARY, WRITTEN, documents=[('a', PRIMARY)],
        extra={'name': 'Injected Person', 'email': 'injected@example.test',
               'identity': {'name': 'Injected Person'}})
    parsed = parse_resume_payload(result['approved_export_payload'])
    assert not parsed.name and not parsed.email


# ── 3. Conflicting supplemental documents ─────────────────────────────────

@requires_pdf
def test_supplemental_documents_never_supply_identity(client, monkeypatch, tmp_path):
    """A second upload's header is not the candidate's unless the caller says so.

    Here the caller designates identity explicitly and the supplement carries a
    different header. The signed identity is the supplied one, and the
    supplement's header text never reaches either deliverable.
    """
    supplement = 'Other Person\nother@example.test\n' + SUPPLEMENT
    source = '\n\n---\n\n'.join([PRIMARY, supplement])
    result, _ = analyze_identity(
        client, monkeypatch, source, PRIMARY,
        documents=[('a', PRIMARY), ('b', supplement)],
        extra={'candidate_identity': IDENTITY})

    assert result['processing_status'] == 'completed', result
    parsed = parse_resume_payload(result['approved_export_payload'])
    assert parsed.name == IDENTITY['name']

    docx_text, pdf_text = rendered_texts(client, result, tmp_path)
    for text in (docx_text, pdf_text):
        assert 'Other Person' not in text
        assert 'other@example.test' not in text
        assert 'Calderon' in text


@requires_pdf
def test_conflicting_supplement_without_supplied_identity_stays_empty(client, monkeypatch, tmp_path):
    """With no designated identity, disagreement between documents is not resolved
    by picking one — the header stays empty."""
    supplement = 'Other Person\nother@example.test\n' + SUPPLEMENT
    source = '\n\n---\n\n'.join([PRIMARY, supplement])
    result, _ = analyze_identity(
        client, monkeypatch, source, PRIMARY,
        documents=[('a', PRIMARY), ('b', supplement)])

    parsed = parse_resume_payload(result['approved_export_payload'])
    assert not parsed.name and not parsed.email
    docx_text, pdf_text = rendered_texts(client, result, tmp_path)
    for text in (docx_text, pdf_text):
        assert 'Other Person' not in text and 'other@example.test' not in text


# ── 4. Tampering ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('field', ['name', 'email', 'phone', 'location',
                                   'linkedin', 'github'])
def test_identity_tampering_is_refused_on_export(client, monkeypatch, field):
    result, _ = analyze_identity(
        client, monkeypatch, PRIMARY, WRITTEN, documents=[('a', PRIMARY)],
        extra={'candidate_identity': IDENTITY})
    assert result['processing_status'] == 'completed', result

    tampered = copy.deepcopy(result)
    tampered['approved_export_payload'][field] = 'forged-value'
    assert export(client, tampered).status_code == 409


def test_clearing_a_signed_identity_is_refused_on_export(client, monkeypatch):
    """Removing a name the approval signed is a content change, not a no-op."""
    result, _ = analyze_identity(
        client, monkeypatch, PRIMARY, WRITTEN, documents=[('a', PRIMARY)],
        extra={'candidate_identity': IDENTITY})
    tampered = copy.deepcopy(result)
    tampered['approved_export_payload']['name'] = ''
    assert export(client, tampered).status_code == 409


def test_adding_identity_to_an_empty_header_is_refused_on_export(client, monkeypatch):
    """The reverse direction: a signed empty header cannot be filled in later."""
    result, _ = analyze_identity(client, monkeypatch, PRIMARY, WRITTEN,
                                 documents=[('a', PRIMARY)])
    tampered = copy.deepcopy(result)
    tampered['approved_export_payload']['name'] = 'Added Later'
    assert export(client, tampered).status_code == 409


def test_another_owner_cannot_export_the_identity_bound_resume(client, monkeypatch):
    result, _ = analyze_identity(
        client, monkeypatch, PRIMARY, WRITTEN, documents=[('a', PRIMARY)],
        extra={'candidate_identity': IDENTITY})
    body = {
        'final_written_section_markdown': result['final_written_section_markdown'],
        'publication_receipt': result['publication_receipt'],
        'skill_inference_policy_state': result['skill_inference_policy_state'],
        **result['approved_export_payload'],
    }
    assert client.post('/resume/export', headers={'X-Cogito-Owner-Id': 'other'},
                       json=body).status_code == 409


# ── 5. Request-model validation ───────────────────────────────────────────

@pytest.mark.parametrize('bad', [
    {'name': 'Jeff', 'nickname': 'Jeff'},          # unknown field
    {'name': 123},                                  # wrong type
    {'name': 'Jeff\nCalderon'},                     # multi-line
    {'name': 'x' * 201},                            # over-long
    'not-an-object',
])
def test_malformed_identity_is_a_validation_error(bad):
    from pydantic import ValidationError
    from models import AnalysisRequest

    with pytest.raises(ValidationError):
        AnalysisRequest(resume_text='x', job_ad='y', candidate_identity=bad)


def test_identity_is_normalised_and_blank_fields_dropped():
    from models import AnalysisRequest

    request = AnalysisRequest(
        resume_text='x', job_ad='y',
        candidate_identity={'name': '  Jeff Calderon  ', 'email': '   ',
                            'phone': None})
    assert request.candidate_identity == {'name': 'Jeff Calderon'}
