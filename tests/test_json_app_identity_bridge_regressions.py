"""Offline JSON route, source identity, bridge and marker regressions."""
import base64
import copy
import io
import shutil
import subprocess

import pytest
from docx import Document

import app
from tests.test_json_bridge_app import client, analyze, export, ScriptedSession, choose, SOURCE, DRAFT, bridge
from tests.test_analyze_source_correction_route import pipeline


HEADER = ('Jane Example\n'
          'jane@example.test | 202-555-0147 | linkedin.com/in/jane-example | github.com/jane-example\n')


def analyze_identity(client, monkeypatch, source, draft, *, documents=None, extra=None):
    session = ScriptedSession([[]])
    app.app.state.http_session = session
    pipeline(monkeypatch, draft=draft, extra={'skill_inference_mode': 'conservative',
        'skill_inference_policy_version': 'conservative-v1', 'skill_bridges': []})
    body = {
        'resume_text': source, 'job_ad': 'Synthetic analyst role',
        'skill_inference_mode': 'conservative', **(extra or {})}
    if documents is not None:
        body['source_documents'] = [
            {'source_id': source_id, 'text': text, 'kind': 'candidate_document'}
            for source_id, text in documents]
    result = client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner'}, json=body).json()
    return result, session


@pytest.mark.parametrize('documents,draft,excluded', [
    ([('a', 'Project Manager\nrecruiter@example.test\n' + SOURCE)], DRAFT,
     ('Project Manager', 'recruiter@example.test')),
    ([('a', 'Project Manager | recruiter@example.test\n' + SOURCE)], DRAFT,
     ('Project Manager', 'recruiter@example.test')),
    ([('a', 'Recruiter Name | recruiter@example.test\n' + SOURCE)], DRAFT,
     ('Recruiter Name', 'recruiter@example.test')),
    ([('a', 'Jane Example\nOther Person\njane@example.test\n' + SOURCE)], DRAFT,
     ('Jane Example', 'Other Person', 'jane@example.test')),
    ([('a', HEADER + SOURCE)], DRAFT,
     ('Jane Example', 'jane@example.test', '202-555-0147', 'jane-example')),
    ([('a', HEADER + SOURCE), ('b', 'Other Person\nother@example.test\n' + SOURCE)], DRAFT,
     ('Jane Example', 'Other Person', 'jane@example.test', 'other@example.test')),
    ([('a', HEADER + SOURCE), ('b', HEADER + SOURCE)], DRAFT,
     ('Jane Example', 'jane@example.test', 'jane-example')),
])
def test_untrusted_headers_not_signed_or_rendered(client, monkeypatch, tmp_path, documents, draft, excluded):
    if not shutil.which('pdflatex') or not shutil.which('pdftotext'):
        pytest.skip('PDF tools unavailable')
    source = '\n\n---\n\n'.join(text for _, text in documents)
    result, session = analyze_identity(client, monkeypatch, source, draft, documents=documents,
        extra={'name': 'Injected Person', 'email': 'injected@example.test',
               'identity': {'name': 'Injected Person'}})
    assert result['processing_status'] == 'completed', result
    payload = result['approved_export_payload']
    from resume_builder.build_resume_from_export import parse_resume_payload
    parsed = parse_resume_payload(payload)
    assert all(not getattr(parsed, field) for field in ('name', 'email', 'phone', 'linkedin', 'github', 'location'))
    assert 'injected@example.test' not in str(session.requests)
    response = export(client, result)
    assert response.status_code == 200, response.text
    rendered = response.json()
    docx = Document(io.BytesIO(base64.b64decode(rendered['docx_base64'])))
    docx_text = '\n'.join(p.text for p in docx.paragraphs)
    pdf = tmp_path / 'resume.pdf'
    pdf.write_bytes(base64.b64decode(rendered['pdf_base64']))
    pdf_text = subprocess.run(['pdftotext', str(pdf), '-'], check=True, capture_output=True, text=True).stdout
    for text in (docx_text, pdf_text):
        assert 'Acme' in text
        assert all(value not in text for value in (*excluded, 'Injected Person', 'injected@example.test'))
    for field in ('name', 'email', 'phone', 'linkedin', 'github'):
        tampered = copy.deepcopy(result)
        tampered['approved_export_payload'][field] = 'forged'
        assert export(client, tampered).status_code == 409
    body = {**payload, 'final_written_section_markdown': result['final_written_section_markdown'],
            'publication_receipt': result['publication_receipt'],
            'skill_inference_policy_state': result['skill_inference_policy_state']}
    assert client.post('/resume/export', headers={'X-Cogito-Owner-Id': 'other'}, json=body).status_code == 409


def test_unprovided_source_documents_do_not_enable_inference(client, monkeypatch):
    result, _ = analyze_identity(client, monkeypatch, HEADER + SOURCE, DRAFT)
    assert result['processing_status'] == 'completed', result
    assert not result['approved_export_payload'].get('name')
    assert not result['approved_export_payload'].get('email')


@pytest.mark.parametrize('documents,header', [
    ([('a', HEADER + SOURCE)], '# Other Person'),
    ([('a', HEADER + SOURCE)], '# Jane Example'),
    ([('a', 'Project Manager\nrecruiter@example.test\n' + SOURCE)], '# Project Manager'),
    ([('a', HEADER + SOURCE), ('b', 'Other Person\nother@example.test\n' + SOURCE)], '# Jane Example'),
])
@pytest.mark.parametrize('sse', [False, True])
def test_generated_header_requires_review_even_after_semantic_approval(client, monkeypatch, documents, header, sse):
    from tests.test_analyze_source_correction_route import result_from_sse

    source = '\n\n---\n\n'.join(text for _, text in documents)
    session = ScriptedSession([[]])
    app.app.state.http_session = session
    pipeline(monkeypatch, draft=header + '\n' + DRAFT, extra={
        'skill_inference_mode': 'conservative',
        'skill_inference_policy_version': 'conservative-v1', 'skill_bridges': []})
    headers = {'X-Cogito-Owner-Id': 'owner'}
    if sse:
        headers['Accept'] = 'text/event-stream'
    response = client.post('/analyze', headers=headers, json={
        'resume_text': source, 'job_ad': 'Synthetic analyst role',
        'skill_inference_mode': 'conservative',
        'source_documents': [{'source_id': sid, 'text': text, 'kind': 'candidate_document'}
                             for sid, text in documents]})
    if sse:
        blocks, result = result_from_sse(response)
        assert not any(block.startswith('event: token') for block in blocks)
    else:
        result = response.json()
    assert len(session.requests) == 1  # The reviewer approved; this gate still vetoes.
    assert result['requires_review'] and not result['retryable']
    assert result['review_reason'] == 'identity_header_unverified'
    assert result['publication_receipt'] is None and result['approved_export_payload'] is None
    assert result['final_written_section_markdown'] is None


@pytest.mark.parametrize('choice,replacement', [('accept', ''), ('remove', ''), ('narrow', 'data analysis')])
def test_unused_bridge_is_validated_but_not_signed(client, monkeypatch, choice, replacement):
    session = ScriptedSession([choose(choice, replacement)])
    result = analyze(client, monkeypatch, session, draft='## Summary\n' + DRAFT.replace('- trend analysis\n', ''))
    assert result['processing_status'] == 'completed', result
    assert result['skill_inference_policy_state']['surviving_bridges'] == []
    assert len(session.requests) == 1
    assert export(client, result).status_code == 200


def test_accepted_embedded_bridge_is_unchanged_and_signed(client, monkeypatch):
    draft = DRAFT.replace('- trend analysis', '- Applied trend  analysis to weekly sales')
    session = ScriptedSession([choose('accept')])
    result = analyze(client, monkeypatch, session, draft=draft)
    assert result['processing_status'] == 'completed', result
    assert result['final_written_section_markdown'] == draft
    assert result['skill_inference_policy_state']['surviving_bridges'][0]['skill'] == 'trend analysis'
    assert len(session.requests) == 1
    assert export(client, result).status_code == 200
    bad = copy.deepcopy(result)
    bad['skill_inference_policy_state']['surviving_bridges'] = []
    assert export(client, bad).status_code == 409


@pytest.mark.parametrize('choice,replacement', [('remove', ''), ('narrow', 'data analysis')])
def test_embedded_bridge_edits_remain_blocked(client, monkeypatch, choice, replacement):
    draft = DRAFT.replace('- trend analysis', '- Applied trend analysis to weekly sales')
    result = analyze(client, monkeypatch, ScriptedSession([choose(choice, replacement)]), draft=draft)
    assert result['requires_review'] and not result['publication_receipt']


@pytest.mark.parametrize('marker', ['( omitted )', '[ OMIT ]', '(\tomit\t)', '[\nomitted\n]'])
def test_marker_whitespace_cannot_be_approved(client, monkeypatch, marker):
    result = analyze(client, monkeypatch, ScriptedSession([[]]), mode='conservative',
                     draft=DRAFT.replace('- trend analysis', '- ' + marker), bridges=[])
    assert result['requires_review'] and not result['publication_receipt']


def test_unrelated_omitted_word_is_not_a_marker(client, monkeypatch):
    draft = DRAFT.replace('- trend analysis', '- omitted details are recorded')
    result = analyze(client, monkeypatch, ScriptedSession([[]]), mode='conservative', draft=draft, bridges=[])
    assert result['processing_status'] == 'completed', result


def test_unused_bridge_still_requires_a_valid_decision(client, monkeypatch):
    draft = '## Summary\n' + DRAFT.replace('- trend analysis\n', '')
    result = analyze(client, monkeypatch, ScriptedSession([choose('accept', 'data analysis')]), draft=draft)
    assert result['requires_review'] and not result['publication_receipt']


def test_mixed_embedded_accept_and_standalone_narrow_are_rereviewed(client, monkeypatch):
    from skill_bridge import build_bridges
    from tests.test_json_bridge_app import QUOTE

    candidates, rejected = build_bridges([
        {'skill': 'trend analysis', 'source_id': 'resume', 'evidence_quote': QUOTE, 'rationale': 'General skill.'},
        {'skill': 'team coordination', 'source_id': 'resume', 'evidence_quote': QUOTE, 'rationale': 'General skill.'},
    ], sources={'resume': SOURCE}, mode='imaginative')
    assert len(candidates) == 2 and not rejected
    draft = DRAFT.replace('- reporting\n- trend analysis',
                          '- reporting\n- team coordination\n- Applied trend analysis to weekly sales')
    def decide(doc):
        return [{'id': item['id'], 'decision': 'accept' if item['skill'] == 'trend analysis' else 'narrow',
                 'replacement': '' if item['skill'] == 'trend analysis' else 'data analysis'}
                for item in doc['bridges']]
    session = ScriptedSession([decide, choose('accept')])
    result = analyze(client, monkeypatch, session, draft=draft, bridges=[b.to_dict() for b in candidates])
    assert result['processing_status'] == 'completed', result
    assert 'Applied trend analysis to weekly sales' in result['final_written_section_markdown']
    assert '- data analysis' in result['final_written_section_markdown']
    assert {b['skill'] for b in result['skill_inference_policy_state']['surviving_bridges']} == {'trend analysis', 'data analysis'}
    assert len(session.requests) == 2


def test_marker_in_source_record_cannot_be_restored():
    from tests.test_semantic_json_engine import Reviewer, answer, insert, views, run, SOURCE as SMALL_SOURCE, DRAFT as SMALL_DRAFT

    source = SMALL_SOURCE.replace('- Processed invoices.', '- Processed invoices. [ omitted ]')
    draft = SMALL_DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '')
    def restore(payload):
        doc, src = views(payload)
        return answer('revise', [insert(doc['sections'][0]['id'], src[0]['records'][1]['id'])])
    reviewer = Reviewer(restore)
    result = run(reviewer, draft=draft, sources={'one': source})
    assert result.status == 'withheld' and result.repair_calls == 0 and len(reviewer.calls) == 1
