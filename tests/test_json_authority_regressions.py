"""Offline JSON-authority, budget, and route-to-export regressions."""
import copy
import json
import time
from unittest.mock import Mock

import pytest

import app
import correction_provider as cp
from semantic_review import json_engine as engine
from tests.test_json_bridge_app import (  # fixtures and transport remain network-disabled
    client, analyze, export, ScriptedSession, choose, SOURCE, DRAFT, PRIVATE,
)
from tests.test_semantic_json_engine import (Reviewer, answer, insert, delete, views, run,
                                              SOURCE as SMALL_SOURCE, DRAFT as SMALL_DRAFT)
from tests.test_correction_integration_offline import Response
from tests.test_analyze_source_correction_route import pipeline
from tests.test_semantic_provider_adapter import Session, envelope, ANSWER, MODEL
from pipeline_config import OR_SLUG_DRAFTER


def test_distinct_missing_records_are_restored_once_in_review_order():
    source = SMALL_SOURCE.replace('## Projects', '**Gamma** -- City\n- Filed forms.\n## Projects')
    draft = SMALL_DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '')
    def batch(payload):
        doc, src = views(payload)
        return answer('revise', [insert(doc['sections'][0]['id'], src[0]['records'][1]['id']),
                                 insert(doc['sections'][0]['id'], src[0]['records'][2]['id'])])
    reviewer = Reviewer(batch, answer())
    result = run(reviewer, draft=draft, sources={'one': source})
    assert result.status == 'approved' and result.restore_calls == 2
    assert result.markdown.index('**Beta**') < result.markdown.index('**Gamma**') < result.markdown.index('## Projects')
    assert views(reviewer.calls[-1][1])[0]['document'] == result.markdown


def test_duplicate_insertion_is_a_conflict_not_two_restorations():
    draft = SMALL_DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '')
    def duplicate(payload):
        doc, src = views(payload)
        item = insert(doc['sections'][0]['id'], src[0]['records'][1]['id'])
        return answer('revise', [item, copy.deepcopy(item)])
    result = run(Reviewer(duplicate), draft=draft)
    assert result.status == 'withheld' and result.repair_calls == 0


def passage_insert(destination, source, passage):
    return {'operation': 'insert_passage', 'destination': destination, 'sources': [source],
            'source_passage': passage, 'problem': 'omitted', 'why': 'missing fact'}


@pytest.mark.parametrize('reverse', [False, True])
def test_omitted_source_passage_and_other_edit_share_safe_boundary(reverse):
    source = SMALL_SOURCE.replace('**Acme** -- City\n', '**Acme** -- City\n*Manager | 2020 - 2022*\n')
    draft = SMALL_DRAFT.replace('**Acme** -- City\n', '**Acme** -- City\n*Manager | 2020 - 2022*\n')
    def batch(payload):
        doc, src = views(payload)
        acme, source = doc['records'][0], src[0]['records'][0]
        missing = passage_insert(acme['id'], source['id'], source['passages'][1]['id'])
        remove = delete(acme['passages'][0]['id'], source['id'])
        return answer('revise', [missing, remove] if reverse else [remove, missing])
    reviewer = Reviewer(batch, answer())
    result = run(reviewer, draft=draft, sources={'one': source})
    assert result.status == 'approved' and result.restore_calls == 1
    assert '*Manager | 2020 - 2022*\n- Tracked orders.\n**Beta**' in result.markdown
    assert views(reviewer.calls[-1][1])[0]['document'] == result.markdown


@pytest.mark.parametrize('mutation', ['wrong_owner', 'wrong_source_passage', 'unknown_destination',
                                      'duplicate', 'existing', 'protected', 'no_role_date',
                                      'wrong_role', 'wrong_date', 'second_refuses'])
def test_omitted_passage_never_bypasses_ownership_or_fresh_review(mutation):
    source = SMALL_SOURCE.replace('**Acme** -- City\n', '**Acme** -- City\n*Manager | 2020 - 2022*\n')
    draft = SMALL_DRAFT.replace('**Acme** -- City\n', '**Acme** -- City\n*Manager | 2020 - 2022*\n')
    if mutation == 'existing':
        draft = draft.replace('**Beta**', '- Tracked orders.\n**Beta**')
    if mutation == 'protected':
        source = source.replace('Tracked orders.', 'Tracked orders. private@example.test')
    if mutation == 'no_role_date':
        draft = draft.replace('*Manager | 2020 - 2022*\n', '')
        source = source.replace('*Manager | 2020 - 2022*\n', '')
    if mutation == 'wrong_role': draft = draft.replace('Manager |', 'Supervisor |')
    if mutation == 'wrong_date': draft = draft.replace('2020 - 2022', '2020 - 2023')
    def proposal(payload):
        doc, src = views(payload)
        record, source_record = doc['records'][0], src[0]['records'][0]
        item = passage_insert(record['id'], source_record['id'], source_record['passages'][1]['id'])
        if mutation == 'wrong_owner': item['destination'] = doc['records'][1]['id']
        if mutation == 'wrong_source_passage': item['source_passage'] = src[0]['records'][1]['passages'][0]['id']
        if mutation == 'unknown_destination': item['destination'] = 'forged'
        return answer('revise', [item, copy.deepcopy(item)] if mutation == 'duplicate' else [item])
    reviewer = Reviewer(proposal, answer('cannot_assess'))
    result = run(reviewer, draft=draft, sources={'one': source})
    assert result.status == 'withheld' and result.markdown is None
    assert len(reviewer.calls) == (2 if mutation == 'second_refuses' else 1)


def test_omitted_passage_schema_is_identical_at_engine_and_provider():
    item = passage_insert('draft-record', 'source-record', 'source-passage')
    response = answer('revise', [item])
    assert engine._shape(response)
    assert cp._validate_semantic_response(response) is None
    variants = cp.SEMANTIC_REVIEW_SCHEMA['properties']['issues']['items']['oneOf']
    variant = next(v for v in variants if v['properties']['operation']['enum'] == ['insert_passage'])
    assert set(variant['required']) == set(item)


@pytest.mark.parametrize('reverse', [False, True])
def test_delete_last_bullet_and_insert_at_adjacent_boundary(reverse):
    draft = SMALL_DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '')
    def batch(payload):
        doc, src = views(payload)
        edits = [delete(doc['records'][0]['passages'][0]['id'], src[0]['records'][0]['id']),
                 insert(doc['sections'][0]['id'], src[0]['records'][1]['id'])]
        return answer('revise', list(reversed(edits)) if reverse else edits)
    result = run(Reviewer(batch, answer()), draft=draft)
    assert result.status == 'approved'
    assert '**Acme** -- City\n**Beta** -- City\n- Processed invoices.\n## Projects' in result.markdown


def test_config_cannot_expand_http_budget_above_four():
    with pytest.raises(ValueError):
        cp.CorrectionProvider('fake', object(), semantic_max_http_attempts=5)


def test_transport_and_format_retry_share_four_post_limit(monkeypatch):
    monkeypatch.setenv(cp.DISCLOSURE_SETTING, '1')
    async def instant(_):
        pass
    monkeypatch.setattr(cp, '_sleep', instant)
    session = Session([(503, ''), (200, envelope('not json')),
                       (503, ''), (200, envelope('still not json')),
                       (200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider('fake', session, model=MODEL)
    result = run(provider, draft=SMALL_SOURCE)
    assert result.reason == 'malformed_output'
    assert provider.attempts == len(session.requests) == 4
    assert provider.usage['total_tokens'] == 30
    assert len(provider.usage['failures']) >= 2
    payloads = [json.loads(r['messages'][1]['content']) for r in session.requests]
    assert payloads[0] == payloads[1]
    assert payloads[2] == payloads[3]
    assert payloads[0]['text'] == payloads[2]['text']
    assert payloads[0]['sources'] == payloads[2]['sources']
    assert payloads[2]['contract'].removeprefix(payloads[0]['contract']) == (
        '\nPrevious response had invalid JSON shape. Return only an object matching the specified schema.')


def test_deadline_checks_after_response_validation(monkeypatch):
    monkeypatch.setenv(cp.DISCLOSURE_SETTING, '1')
    clock = [100.0]
    monkeypatch.setattr(cp.time, 'monotonic', lambda: clock[0])
    validate = cp._validate_semantic_response
    def slow(answer):
        validate(answer)
        clock[0] = 104.0
    monkeypatch.setattr(cp, '_validate_semantic_response', slow)
    session = Session([(200, envelope(json.dumps(ANSWER)))])
    provider = cp.CorrectionProvider('fake', session, model=MODEL)
    provider.set_budget(3)
    from tests.test_semantic_provider_adapter import run as provider_run
    with pytest.raises(cp.ProviderCallTimeout):
        provider_run(provider)
    assert len(session.requests) == 1 and provider.usage['total_tokens'] == 15


def test_deadline_includes_synchronous_source_validation(monkeypatch):
    original = engine._snapshot
    def slow(sources):
        time.sleep(0.02)
        return original(sources)
    monkeypatch.setattr(engine, '_snapshot', slow)
    reviewer = Reviewer(answer())
    result = run(reviewer, draft=SMALL_SOURCE, timeout_seconds=0.005)
    assert result.reason == 'timeout_initial_review' and not reviewer.calls


def test_bridge_policy_disallows_unlisted_inferences():
    payload, *_ = engine._payload(SMALL_SOURCE, (('resume', SMALL_SOURCE),), 'imaginative', [{
        'id': 'bridge', 'skill': 'analysis', 'source_id': 'resume', 'quote_start': 0, 'quote_end': 2}])
    assert 'Only the individually accepted or narrowed bridge labels' in payload['contract']
    assert 'third-party claims' in payload['contract'] and 'generated text' in payload['contract']


class RevisionSession(ScriptedSession):
    def __init__(self):
        super().__init__([])
    def post(self, url, **kwargs):
        self.requests.append(copy.deepcopy(kwargs['json']))
        payload = json.loads(kwargs['json']['messages'][1]['content'])
        doc, src = json.loads(payload['text']), json.loads(payload['sources'][0]['text'])
        if len(self.requests) == 1:
            target = next(r for r in doc['records'] if r['owner'] == 'Acme')
            original = next(r for r in src['records'] if r['owner'] == 'Acme')
            reply = answer('revise', [{'operation': 'replace', 'passage': target['passages'][1]['id'],
                        'problem': 'unsupported', 'sources': [original['id']], 'why': 'unsupported',
                        'correction': 'Documented weekly sales activity and coordinated reporting with team members, reviewed written records and shared the information with colleagues during regular meetings about daily activities and operational needs.'}])
        else:
            reply = answer()
        body = {'model': OR_SLUG_DRAFTER, 'usage': {'prompt_tokens': 10, 'completion_tokens': 4,
                'total_tokens': 14}, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(reply)}}]}
        return Response(json.dumps(body).encode())


class OmissionSession(ScriptedSession):
    def __init__(self):
        super().__init__([])
    def post(self, url, **kwargs):
        self.requests.append(copy.deepcopy(kwargs['json']))
        payload = json.loads(kwargs['json']['messages'][1]['content'])
        doc, src = json.loads(payload['text']), json.loads(payload['sources'][0]['text'])
        if len(self.requests) == 1:
            target = next(r for r in doc['records'] if r['owner'] == 'Acme')
            original = next(r for r in src['records'] if r['owner'] == 'Acme')
            reply = answer('revise', [passage_insert(target['id'], original['id'],
                                                       original['passages'][1]['id'])])
        else:
            reply = answer()
        body = {'model': OR_SLUG_DRAFTER, 'usage': {'prompt_tokens': 10, 'completion_tokens': 4,
                'total_tokens': 14}, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(reply)}}]}
        return Response(json.dumps(body).encode())


def test_route_restores_omitted_source_passage_and_exports_it(client, monkeypatch):
    source = SOURCE.replace(PRIVATE, 'the client').replace('**Acme** -- City\n',
        '**Acme** -- City\n*Manager | 2020 - 2022*\n')
    omitted = next(line for line in source.splitlines() if line.startswith('- Documented weekly'))
    draft = source.replace(omitted + '\n', '').replace('- reporting\n',
        '- reporting, collaboration, communication, scheduling, planning, customer service, '
        'documentation, time management, problem solving, team coordination, quality assurance, '
        'information review, workflow management, organization and administration, '
        'business analysis and client service\n')
    pipeline(monkeypatch, draft=draft, extra={'skill_inference_mode': 'conservative',
        'skill_inference_policy_version': 'conservative-v1', 'skill_bridges': []})
    session = OmissionSession()
    app.app.state.http_session = session
    result = client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner'}, json={
        'resume_text': source, 'job_ad': 'Synthetic role', 'skill_inference_mode': 'conservative'}).json()
    assert result['processing_status'] == 'completed', result
    assert len(session.requests) == 2
    assert omitted in result['final_written_section_markdown']
    assert omitted[2:] in result['approved_export_payload']['experience'][0]['bullets']
    renderer = Mock(return_value={'filename': 'test', 'pdf_base64': 'AA', 'docx_base64': 'AA'})
    monkeypatch.setattr('resume_builder.build_resume_from_export.build_resume_from_payload_base64', renderer)
    assert export(client, result).status_code == 200
    assert renderer.call_count == 1


def test_real_route_repair_rereview_exact_export_and_identity(client, monkeypatch):
    session = RevisionSession()
    sentence = 'regular meetings about daily activities and operational needs.'
    source = SOURCE.replace(PRIVATE, 'the client').replace('- reporting\n', '- reporting\n- ' + PRIVATE + '\n')
    draft = source.replace(sentence, sentence + ' Used a fictional system.')
    app.app.state.http_session = session
    pipeline(monkeypatch, draft=draft, extra={'skill_inference_mode': 'conservative',
        'skill_inference_policy_version': 'conservative-v1', 'skill_bridges': []})
    result = client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner'}, json={
        'resume_text': source, 'job_ad': 'Synthetic analyst role',
        'skill_inference_mode': 'conservative'}).json()
    assert result['processing_status'] == 'completed', result
    assert len(session.requests) == 2
    reviewed = json.loads(json.loads(session.requests[1]['messages'][1]['content'])['text'])['document']
    assert reviewed == engine._clean(result['final_written_section_markdown'])
    assert 'fictional system' not in result['final_written_section_markdown']
    assert PRIVATE not in json.dumps(session.requests)
    renderer = Mock(return_value={'filename': 'test', 'pdf_base64': 'AA', 'docx_base64': 'AA'})
    monkeypatch.setattr('resume_builder.build_resume_from_export.build_resume_from_payload_base64', renderer)
    assert export(client, result).status_code == 200
    assert renderer.call_count == 1
    bad = copy.deepcopy(result)
    bad['approved_export_payload']['name'] = 'Injected Name'
    assert export(client, bad).status_code == 409
    assert renderer.call_count == 1


def test_route_schema_capability_is_explicit(client, monkeypatch):
    monkeypatch.setenv('CORRECTION_SEMANTIC_JSON_SCHEMA_SUPPORTED', 'true')
    session = ScriptedSession([choose('accept')])
    result = analyze(client, monkeypatch, session)
    assert result['processing_status'] == 'completed'
    assert session.requests[0]['response_format']['json_schema']['strict'] is True
    assert session.requests[0]['provider']['require_parameters'] is True


def test_route_withholds_when_receipt_finishes_after_deadline(client, monkeypatch):
    monkeypatch.setattr(app, 'correction_budget_seconds', lambda *args: 0.03)
    original = app._issue_publication_receipt
    def slow(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)
    monkeypatch.setattr(app, '_issue_publication_receipt', slow)
    session = ScriptedSession([choose('accept')])
    result = analyze(client, monkeypatch, session)
    assert result['requires_review'] and not result['publication_receipt']
    assert result['correction_reason'] == 'timeout_initial_review'


def test_route_final_checks_still_veto_json_approval(client, monkeypatch):
    draft = DRAFT.replace('regular meetings', '99% of regular meetings')
    session = ScriptedSession([choose('accept')])
    result = analyze(client, monkeypatch, session, draft=draft)
    assert len(session.requests) == 1
    assert result['requires_review'] and not result['publication_receipt']


def test_route_requires_owner_even_after_json_approval(client, monkeypatch):
    session = ScriptedSession([choose('accept')])
    app.app.state.http_session = session
    from tests.test_json_bridge_app import bridge
    pipeline(monkeypatch, draft=DRAFT, extra={'skill_inference_mode': 'imaginative',
        'skill_inference_policy_version': 'imaginative-v1', 'skill_bridges': [bridge()]})
    result = client.post('/analyze', json={'resume_text': SOURCE,
        'job_ad': 'Synthetic role', 'skill_inference_mode': 'imaginative'}).json()
    assert result['requires_review'] and result['correction_reason'] == 'receipt_unavailable'
    assert not result['publication_receipt'] and not result['approved_export_payload']


def test_route_source_section_missing_reports_original_reason(client, monkeypatch):
    # Move the education body to Projects, preserving a long renderable draft.
    draft = DRAFT.replace('## Education\n**Synthetic College**\n',
                          '## Education\n## Projects\n**Synthetic College**\n')
    session = ScriptedSession([[]])
    result = analyze(client, monkeypatch, session, mode='conservative', draft=draft, bridges=[])
    assert result['requires_review'] and not result['publication_receipt']
    assert result['correction_reason'] == 'source_section_missing'


def test_route_rejects_untrusted_or_mismatched_source_documents(client, monkeypatch):
    session = ScriptedSession([])
    app.app.state.http_session = session
    pipeline(monkeypatch, draft=DRAFT)
    for documents in ([{'source_id': 'one', 'text': SOURCE, 'kind': 'job_ad'}],
                      [{'source_id': 'one', 'text': SOURCE + ' changed', 'kind': 'candidate_document'}]):
        result = client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner'}, json={
            'resume_text': SOURCE, 'source_documents': documents, 'job_ad': 'Synthetic role'}).json()
        assert result['review_reason'] == 'source_documents_mismatch'
        assert not result['publication_receipt'] and not session.requests


def test_route_reviews_all_ordered_candidate_documents_not_job_ad(client, monkeypatch):
    first = SOURCE
    second = '## Professional Experience\n**Beta** -- City\n- Processed inventory.\n'
    source = first + '\n\n---\n\n' + second
    draft = DRAFT.replace('## Projects', '**Beta** -- City\n- Processed inventory.\n## Projects')
    pipeline(monkeypatch, draft=draft, extra={'skill_inference_mode': 'conservative',
        'skill_inference_policy_version': 'conservative-v1', 'skill_bridges': []})
    session = ScriptedSession([[]])
    app.app.state.http_session = session
    result = client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner'}, json={
        'resume_text': source, 'job_ad': 'ONLY JOB AD NOT EVIDENCE', 'skill_inference_mode': 'conservative',
        'source_documents': [{'source_id': 'one', 'text': first, 'kind': 'candidate_document'},
                             {'source_id': 'two', 'text': second, 'kind': 'candidate_document'}]}).json()
    assert result['processing_status'] == 'completed', result
    wire = json.loads(session.requests[0]['messages'][1]['content'])
    assert [json.loads(s['text'])['document'] for s in wire['sources']] == [engine._clean(first), second]
    assert 'ONLY JOB AD NOT EVIDENCE' not in json.dumps(wire)
    assert len({s['source_id'] for s in wire['sources']}) == 2
