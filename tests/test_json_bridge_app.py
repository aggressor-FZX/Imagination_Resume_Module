"""Offline route-to-export bridge lifecycle using the real provider adapter."""
import copy
import json
import socket
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import app
from pipeline_config import OR_SLUG_DRAFTER
from skill_bridge import build_bridges
from tests.test_correction_integration_offline import Response, indexed_body_inventory, indexed_body_review
from tests.test_analyze_source_correction_route import pipeline


PRIVATE = 'person@example.test'
QUOTE = f'Compared weekly sales for {PRIVATE} to identify trends.'
SOURCE = (f'## Professional Experience\n**Acme** -- City\n- {QUOTE}\n'
          '- Documented weekly sales activity and coordinated reporting with team members, '
          'reviewed written records and shared the information with colleagues during '
          'regular meetings about daily activities and operational needs.\n'
          '## Projects\n## Education\n**Synthetic College**\n'
          '- Completed technical training in reporting and business operations with '
          'practical instruction, supervised exercises and written assignments about '
          'collaboration, communication, record keeping and routine workplace procedures.\n'
          '## Certifications\n## Technical Skills\n- reporting\n')
DRAFT = SOURCE.replace('- reporting', '- reporting\n- trend analysis')


def bridge():
    accepted, rejected = build_bridges([{
        'skill': 'trend analysis', 'source_id': 'resume', 'evidence_quote': QUOTE,
        'rationale': f'Private rationale {PRIVATE}',
    }], sources={'resume': SOURCE}, mode='imaginative')
    assert len(accepted) == 1 and not rejected
    return accepted[0].to_dict()


class ScriptedSession:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.closed = False

    async def close(self):
        self.closed = True

    def post(self, url, **kwargs):
        self.requests.append(copy.deepcopy(kwargs['json']))
        payload = json.loads(kwargs['json']['messages'][1]['content'])
        if 'text' in payload:
            doc = json.loads(payload['text'])
            choices = self.decisions.pop(0)
            if callable(choices):
                choices = choices(doc)
            response = {'version': 'semantic_review_v1', 'decision': 'approve',
                        'rationale': '', 'issues': [], 'bridge_decisions': choices}
        elif 'lines' in payload:
            response = indexed_body_review(payload)
            if payload.get('bridges'):
                response['bridge_decisions'] = {b['id']: 'accept' for b in payload['bridges']}
                for b in payload['bridges']:
                    anchor = next(s['index'] for s in payload['segments'] if b['evidence'] in s['text'])
                    for line in payload['lines']:
                        if line['text'].strip().lstrip('- ').strip() == b['skill']:
                            lid = line['line_id']
                            for key in ('supported', 'unsupported', 'uncertain', 'structural'):
                                if lid in response[key]:
                                    response[key].remove(lid)
                            response['supported'].append(lid)
                            response['evidence'][lid] = [anchor]
        else:
            response = indexed_body_inventory(payload)
        body = {'model': OR_SLUG_DRAFTER,
                'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response)}}]}
        return Response(json.dumps(body).encode())


def choose(decision, replacement=''):
    def answer(doc):
        return [{'id': item['id'], 'decision': decision, 'replacement': replacement}
                for item in doc.get('bridges', [])]
    return answer


@pytest.fixture
def client(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr(app.settings, 'openrouter_api_key_1', 'offline')
    monkeypatch.setattr(app.settings, 'openrouter_api_key_2', None)
    monkeypatch.setenv('PUBLICATION_RECEIPT_SECRET', 'offline-secret-01234567890123456789')
    monkeypatch.setenv('CORRECTION_PROVIDER_DISCLOSURE', '1')
    monkeypatch.setenv('CORRECTION_ENGINE', 'semantic_json')
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key, lambda: 'offline')
    with TestClient(app.app) as test_client:
        test_client.portal.call(app.app.state.http_session.close)
        yield test_client
    app.app.dependency_overrides.clear()


def analyze(client, monkeypatch, session, *, mode='imaginative', draft=DRAFT, bridges=None):
    app.app.state.http_session = session
    pipeline(monkeypatch, draft=draft, extra={
        'skill_inference_mode': mode,
        'skill_inference_policy_version': f'{mode}-v1',
        'skill_bridges': [bridge()] if bridges is None else bridges,
    })
    return client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner'},
                     json={'resume_text': SOURCE, 'job_ad': 'Synthetic analyst role',
                         'skill_inference_mode': mode}).json()


def export(client, result):
    return client.post('/resume/export', headers={'X-Cogito-Owner-Id': 'owner'}, json={
        **result['approved_export_payload'],
        'final_written_section_markdown': result['final_written_section_markdown'],
        'publication_receipt': result['publication_receipt'],
        'skill_inference_policy_state': result['skill_inference_policy_state'],
    })


@pytest.mark.parametrize('choice,followup,expected', [
    ('accept', None, 'trend analysis'),
    ('narrow', 'data analysis', 'data analysis'),
    ('remove', None, None),
])
def test_bridge_lifecycle_to_export(client, monkeypatch, choice, followup, expected):
    session = ScriptedSession([choose(choice, followup or ''), choose('accept')])
    result = analyze(client, monkeypatch, session)
    assert result['processing_status'] == 'completed', (result['review_reasons'], result['correction_reason'], len(session.requests))
    state = result['skill_inference_policy_state']
    assert state['skill_inference_mode'] == 'imaginative'
    assert len(state['surviving_bridges']) == (1 if expected else 0)
    if expected:
        record = state['surviving_bridges'][0]
        assert record['skill'] == expected
        assert set(record) == {'id', 'skill', 'origin', 'source_id', 'policy_version'}
    docs = [json.loads(payload['text']) for r in session.requests
            if 'text' in (payload := json.loads(r['messages'][1]['content']))]
    assert len(docs) == (1 if choice == 'accept' else 2)
    assert len(session.requests) == (1 if choice == 'accept' else 2)
    assert all('text' in json.loads(r['messages'][1]['content']) for r in session.requests)
    assert PRIVATE not in json.dumps(session.requests)
    assert 'Private rationale' not in json.dumps(session.requests)
    assert PRIVATE not in json.dumps(result['skill_inference_policy_state'])
    if choice != 'accept':
        assert docs[0]['document'] != docs[1]['document']
        assert ('trend analysis' not in docs[1]['document'])
        assert not docs[1].get('bridges') if choice == 'remove' else docs[1]['bridges'][0]['skill'] == expected
    renderer = Mock(return_value={'filename': 'test', 'pdf_base64': 'AA', 'docx_base64': 'AA'})
    monkeypatch.setattr('resume_builder.build_resume_from_export.build_resume_from_payload_base64', renderer)
    assert export(client, result).status_code == 200
    renderer.assert_called_once()
    renderer.reset_mock()
    bad = copy.deepcopy(result)
    bad['skill_inference_policy_state']['skill_inference_mode'] = 'conservative'
    assert export(client, bad).status_code == 409
    bad = copy.deepcopy(result)
    bad['skill_inference_policy_state']['surviving_bridges'] = [] if expected else [{'id': 'forged'}]
    assert export(client, bad).status_code == 409
    bad = copy.deepcopy(result)
    bad['approved_export_payload']['experience'][0]['bullets'][0] += ' invented'
    assert export(client, bad).status_code == 409
    renderer.assert_not_called()


def test_conservative_has_no_bridge_and_signs_mode(client, monkeypatch):
    session = ScriptedSession([[]])
    result = analyze(client, monkeypatch, session, mode='conservative',
                     draft='## Summary\n' + SOURCE, bridges=[])
    assert result['processing_status'] == 'completed', (result['review_reasons'], result['correction_reason'], len(session.requests))
    assert result['skill_inference_policy_state']['skill_inference_mode'] == 'conservative'
    assert result['skill_inference_policy_state']['surviving_bridges'] == []
    wire = json.loads(session.requests[0]['messages'][1]['content'])
    assert 'bridges' not in json.loads(wire['text']) and 'CONSERVATIVE' in wire['contract']
    monkeypatch.setattr('resume_builder.build_resume_from_export.build_resume_from_payload_base64',
                        Mock(return_value={'filename': 'test', 'pdf_base64': 'AA', 'docx_base64': 'AA'}))
    assert export(client, result).status_code == 200


@pytest.mark.parametrize('answer', [[], [{'id': 'forged', 'decision': 'accept', 'replacement': ''}],
                                  [{'id': 'bridge-01-fake', 'decision': 'accept', 'replacement': ''}]])
def test_missing_or_forged_decisions_withhold(client, monkeypatch, answer):
    result = analyze(client, monkeypatch, ScriptedSession([answer]))
    assert result['requires_review'] and not result['publication_receipt']


def test_forged_anchor_and_conservative_bridge_withhold_before_call(client, monkeypatch):
    forged = bridge()
    forged['quote_end'] -= 1
    session = ScriptedSession([])
    result = analyze(client, monkeypatch, session, bridges=[forged])
    assert result['requires_review'] and not session.requests
    result = analyze(client, monkeypatch, session, mode='conservative', bridges=[bridge()])
    assert result['requires_review'] and not session.requests


def test_invalid_narrow_and_embedded_removal_withhold(client, monkeypatch):
    result = analyze(client, monkeypatch, ScriptedSession([choose('narrow', 'expert analysis')]))
    assert result['requires_review']
    draft = DRAFT.replace('- trend analysis', '- Applied trend analysis using invented tools')
    result = analyze(client, monkeypatch, ScriptedSession([choose('remove')]), draft=draft)
    assert result['requires_review']


def test_explicit_legacy_config_uses_rollback_route(client, monkeypatch):
    from tests.test_correction_integration_offline import FakeSession, DRAFT as LEGACY_DRAFT, SOURCE as LEGACY_SOURCE
    monkeypatch.setenv('CORRECTION_ENGINE', 'legacy')
    session = FakeSession()
    app.app.state.http_session = session
    pipeline(monkeypatch, draft=LEGACY_DRAFT)
    result = client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner'},
                         json={'resume_text': LEGACY_SOURCE, 'job_ad': 'Synthetic role'}).json()
    assert result['processing_status'] == 'completed', result
    assert [c['phase'] for c in result['run_metrics']['calls'] if c['stage'] == 'source_correction'] == [
        'source_inventory', 'initial_review', 'repair', 'post_patch_review']


def test_unset_config_uses_legacy_route(client, monkeypatch):
    monkeypatch.delenv('CORRECTION_ENGINE')
    session = ScriptedSession([choose('accept')])
    result = analyze(client, monkeypatch, session)
    assert result['processing_status'] == 'completed' and result['publication_receipt']
    assert [c['phase'] for c in result['run_metrics']['calls'] if c['stage'] == 'source_correction'] == [
        'source_inventory', 'initial_review']
    assert len(session.requests) == 2
    inventory, review = [json.loads(r['messages'][1]['content']) for r in session.requests]
    assert 'segments' in inventory and 'lines' not in inventory and 'text' not in inventory
    assert 'segments' in review and 'lines' in review and 'text' not in review


def test_json_approval_does_not_call_legacy_reviewer(client, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError('legacy reviewer must not run')
    monkeypatch.setattr(app, 'correct', forbidden)
    session = ScriptedSession([choose('accept')])
    result = analyze(client, monkeypatch, session)
    assert result['processing_status'] == 'completed' and result['publication_receipt']
    assert len(session.requests) == 1


def test_forged_original_quote_with_same_redaction_is_rejected(client, monkeypatch):
    forged = bridge()
    forged['evidence_quote'] = forged['evidence_quote'].replace(PRIVATE, 'otherx@example.test')
    assert len(forged['evidence_quote']) == len(bridge()['evidence_quote'])
    session = ScriptedSession([])
    result = analyze(client, monkeypatch, session, bridges=[forged])
    assert result['requires_review'] and not session.requests


def test_narrowed_receipt_rejects_original_deck(client, monkeypatch):
    result = analyze(client, monkeypatch, ScriptedSession([choose('narrow', 'data analysis'), choose('accept')]))
    assert result['processing_status'] == 'completed'
    state = result['skill_inference_policy_state']
    assert state['surviving_bridges'][0]['skill'] == 'data analysis'
    assert result['skill_bridges'] == state['surviving_bridges']
    original = copy.deepcopy(result)
    original['skill_inference_policy_state']['surviving_bridges'][0]['skill'] = 'trend analysis'
    monkeypatch.setattr('resume_builder.build_resume_from_export.build_resume_from_payload_base64',
                        Mock(return_value={'filename': 'test', 'pdf_base64': 'AA', 'docx_base64': 'AA'}))
    assert export(client, result).status_code == 200
    assert export(client, original).status_code == 409


def test_unknown_engine_fails_closed(client, monkeypatch):
    monkeypatch.setenv('CORRECTION_ENGINE', 'typo')
    session = ScriptedSession([])
    result = analyze(client, monkeypatch, session)
    assert result['requires_review'] and not result['publication_receipt'] and not session.requests


def test_sse_and_export_owner_are_bound(client, monkeypatch):
    from tests.test_analyze_source_correction_route import result_from_sse

    session = ScriptedSession([choose('narrow', 'data analysis'), choose('accept')])
    app.app.state.http_session = session
    pipeline(monkeypatch, draft=DRAFT, extra={
        'skill_inference_mode': 'imaginative',
        'skill_inference_policy_version': 'imaginative-v1',
        'skill_bridges': [bridge()],
    })
    response = client.post('/analyze', headers={'X-Cogito-Owner-Id': 'owner',
                           'Accept': 'text/event-stream'}, json={
                               'resume_text': SOURCE, 'job_ad': 'Synthetic analyst role',
                               'skill_inference_mode': 'imaginative'})
    blocks, result = result_from_sse(response)
    assert result['processing_status'] == 'completed'
    assert result['skill_inference_policy_state']['surviving_bridges'][0]['skill'] == 'data analysis'
    assert not any('trend analysis' in block for block in blocks if block.startswith('event: token'))
    renderer = Mock(return_value={'filename': 'test', 'pdf_base64': 'AA', 'docx_base64': 'AA'})
    monkeypatch.setattr('resume_builder.build_resume_from_export.build_resume_from_payload_base64', renderer)
    body = {**result['approved_export_payload'],
            'final_written_section_markdown': result['final_written_section_markdown'],
            'publication_receipt': result['publication_receipt'],
            'skill_inference_policy_state': result['skill_inference_policy_state']}
    assert client.post('/resume/export', headers={'X-Cogito-Owner-Id': 'other'}, json=body).status_code == 409
    renderer.assert_not_called()