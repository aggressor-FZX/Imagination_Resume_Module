"""Offline checks for the JSON engine's mode and bridge trust boundary."""
import asyncio
import copy
import json

import pytest

import publication_gate
from semantic_review.json_engine import correct
from skill_bridge import build_bridges
from source_correction import SourceDocument


SOURCE = (
    '## Professional Experience\n**Acme** -- City\n'
    '- Compared weekly sales figures to identify trends.\n'
    '## Projects\n## Education\n## Certifications\n## Technical Skills\n'
)
SOURCES = (SourceDocument('document-1', SOURCE),)
QUOTE = 'Compared weekly sales figures to identify trends.'


def proposal():
    return {'skill': 'trend analysis', 'evidence_quote': QUOTE,
            'source_id': 'document-1', 'rationale': 'General analytical capability.'}


def valid_bridge():
    accepted, rejected = build_bridges([proposal()], sources={'document-1': SOURCE},
                                       mode='imaginative')
    assert len(accepted) == 1 and not rejected
    bridge = accepted[0].to_dict()
    assert SOURCE[bridge['quote_start']:bridge['quote_end']] == QUOTE
    return bridge


class Reviewer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, phase, payload):
        self.calls.append(copy.deepcopy(payload))
        response = self.responses.pop(0)
        return response(payload) if callable(response) else response


def answer(decision='approve', issues=None):
    return {'version': 'semantic_review_v1', 'decision': decision,
            'rationale': '', 'issues': issues or []}


def run(reviewer, *, draft=SOURCE, **kwargs):
    return asyncio.run(correct(draft, SOURCES, reviewer, **kwargs))


def test_structurally_valid_bridge_is_not_semantic_approval():
    bridge = valid_bridge()
    # Even a real Phase B record is a proposal. A generic approve response is
    # not an accept/narrow/remove decision against its anchor.
    reviewer = Reviewer([answer()])
    result = run(reviewer, mode='imaginative', permitted_bridges=[bridge])
    assert result.status == 'withheld' and result.reason == 'unresolved_claims'
    assert result.markdown is None and result.review_calls == 1 and len(reviewer.calls) == 1


@pytest.mark.parametrize('mode', ['conservative', 'imaginative'])
@pytest.mark.parametrize('mutation', ['bad_anchor', 'bad_source', 'missing_quote',
                                      'bad_id', 'unvalidated', 'valid'])
def test_nonempty_bridge_decks_never_bypass_review_or_mode(mode, mutation):
    bridge = valid_bridge()
    if mutation == 'bad_anchor': bridge['quote_end'] -= 1
    if mutation == 'bad_source': bridge['source_id'] = 'other-document'
    if mutation == 'missing_quote': bridge.pop('evidence_quote')
    if mutation == 'bad_id': bridge['id'] = ''
    if mutation == 'unvalidated': bridge = {'skill': 'trend analysis'}
    reviewer = Reviewer([answer()])
    result = run(reviewer, mode=mode, permitted_bridges=[bridge])
    assert result.reason == 'unresolved_claims' and result.markdown is None
    assert len(reviewer.calls) == (1 if mutation == 'valid' and mode == 'imaginative' else 0)


@pytest.mark.parametrize('bridges', [{}, (), '', False, 0])
def test_falsey_malformed_bridge_container_is_not_an_empty_deck(bridges):
    reviewer = Reviewer([answer()])
    result = run(reviewer, mode='imaginative', permitted_bridges=bridges)
    assert result.reason == 'invalid_input' and not reviewer.calls


@pytest.mark.parametrize('mode', ['conservative', 'imaginative'])
@pytest.mark.parametrize('bridges', [None, []])
def test_empty_deck_uses_source_only_policy_for_each_mode(mode, bridges):
    reviewer = Reviewer([answer()])
    result = run(reviewer, mode=mode, permitted_bridges=bridges)
    assert result.status == 'approved' and result.markdown == SOURCE
    contract = reviewer.calls[0]['contract']
    assert mode.upper() in contract
    assert 'do not add unstated' in contract
    assert 'capabilities' in contract
    if mode == 'imaginative':
        assert 'no authorised bridges' in contract


def test_source_only_policy_survives_format_retry_and_full_rereview():
    draft = SOURCE.replace(QUOTE, QUOTE + ' using invented equipment')

    def revise(payload):
        document = json.loads(payload['text'])
        source = json.loads(payload['sources'][0]['text'])
        return answer('revise', [{
            'operation': 'replace', 'passage': document['records'][0]['passages'][0]['id'],
            'problem': 'unsupported', 'sources': [source['records'][0]['id']],
            'why': 'unsupported method', 'correction': QUOTE,
        }])

    reviewer = Reviewer([{}, revise, answer()])
    result = run(reviewer, draft=draft, mode='imaginative', permitted_bridges=[])
    assert result.status == 'approved' and result.markdown == SOURCE
    assert (result.review_calls, result.repair_calls) == (3, 1)
    assert all('no authorised bridges' in call['contract'] for call in reviewer.calls)
    assert json.loads(reviewer.calls[-1]['text'])['document'] == SOURCE


def test_receipt_policy_binding_depends_on_actual_surviving_bridge_record():
    bridge = valid_bridge()
    policy = {'skill_inference_mode': 'imaginative',
              'skill_inference_policy_version': 'imaginative-v1',
              'surviving_bridges': [bridge]}
    narrowed = copy.deepcopy(policy)
    narrowed['surviving_bridges'][0]['skill'] = 'reporting'
    removed = copy.deepcopy(policy)
    removed['surviving_bridges'] = []
    assert publication_gate.policy_fingerprint(policy) != publication_gate.policy_fingerprint(narrowed)
    assert publication_gate.policy_fingerprint(policy) != publication_gate.policy_fingerprint(removed)
    # An approved source-only deck is explicitly empty, not the original proposals.
    result = run(Reviewer([answer()]), mode='imaginative')
    assert result.status == 'approved' and result.final_bridges == ()


def test_default_is_imaginative_but_does_not_automatically_authorize_bridges():
    reviewer = Reviewer([answer()])
    result = run(reviewer)
    assert result.status == 'approved'
    assert 'IMAGINATIVE with no authorised bridges' in reviewer.calls[0]['contract']


def test_deadline_cannot_exceed_the_existing_correction_ceiling():
    from source_correction import MAX_TIMEOUT_SECONDS

    reviewer = Reviewer([answer()])
    result = run(reviewer, timeout_seconds=MAX_TIMEOUT_SECONDS + 1)
    assert result.reason == 'invalid_input' and not reviewer.calls


@pytest.mark.parametrize('defect', ['omitted_record', 'wrong_owner', 'invented_tool'])
def test_generic_approval_does_not_prove_source_coverage_or_attribution(defect):
    """Pin the blocker: a document-level verdict has no independently checked citations."""
    source = SOURCE.replace('## Projects', '**Beta** -- City\n- Processed invoices.\n## Projects')
    draft = source
    if defect == 'omitted_record':
        draft = draft.replace('**Beta** -- City\n- Processed invoices.\n', '')
    elif defect == 'wrong_owner':
        draft = draft.replace('- ' + QUOTE, '- Processed invoices.').replace(
            '- Processed invoices.\n## Projects', '- ' + QUOTE + '\n## Projects')
    else:
        draft = draft.replace(QUOTE, QUOTE + ' using ImaginaryTool')
    reviewer = Reviewer([answer()])
    result = asyncio.run(correct(draft, {'candidate': source}, reviewer, mode='imaginative'))
    assert result.status == 'approved' and result.markdown == draft
    assert result.coverage.level == 'semantic_review_not_proof'
    assert result.coverage.claim_count == 0 and result.coverage.supported_count == 0


def test_current_route_cannot_bind_a_narrowed_deck_from_the_original_proposals():
    import app

    original = valid_bridge()
    narrowed = {**original, 'skill': 'reporting'}
    final = SOURCE + '- reporting\n'
    original_state = app._policy_state_from_output({
        'skill_inference_mode': 'imaginative',
        'skill_inference_policy_version': 'imaginative-v1',
        'skill_bridges': app._surviving_bridges([original], final),
    })
    narrowed_state = app._policy_state_from_output({
        'skill_inference_mode': 'imaginative',
        'skill_inference_policy_version': 'imaginative-v1',
        'skill_bridges': app._surviving_bridges([narrowed], final),
    })
    assert original_state['surviving_bridges'] == []
    assert narrowed_state['surviving_bridges'][0]['skill'] == 'reporting'
    assert publication_gate.policy_fingerprint(original_state) != publication_gate.policy_fingerprint(narrowed_state)


# Reuse the real JSON/SSE route with synthetic provider responses and sockets disabled.
from tests.test_analyze_source_correction_route import (  # noqa: E402
    offline, synthetic_disclosure_bypass, pipeline, post, result_from_sse,
)


@pytest.mark.parametrize('sse', [False, True])
def test_analyze_routes_remain_on_existing_review_until_safe_integration(
        offline, synthetic_disclosure_bypass, monkeypatch, sse):
    import app
    import source_correction

    client, session = offline
    pipeline(monkeypatch)
    assert app.correct is source_correction.correct
    response = post(client, sse=sse)
    result = result_from_sse(response)[1] if sse else response.json()
    assert result['processing_status'] == 'completed'
    assert result['publication_receipt'] and result['approved_export_payload']
    phases = [item['phase'] for item in result['run_metrics']['calls']
              if item['stage'] == 'source_correction']
    assert phases == ['source_inventory', 'initial_review', 'repair', 'post_patch_review']
    assert len(session.requests) == 4


def bridge_answer(choice, replacement=''):
    def respond(payload):
        doc = json.loads(payload['text'])
        return {**answer(), 'bridge_decisions': [
            {'id': b['id'], 'decision': choice, 'replacement': replacement}
            for b in doc.get('bridges', [])]}
    return respond


@pytest.mark.parametrize('choice,replacement,expected', [
    ('accept', '', 'trend analysis'), ('narrow', 'data analysis', 'data analysis'),
    ('remove', '', None)])
def test_final_deck_is_reviewed_and_owned(choice, replacement, expected):
    original = valid_bridge()
    draft = SOURCE + '- trend analysis\n'
    reviewer = Reviewer([bridge_answer(choice, replacement), bridge_answer('accept')])
    result = run(reviewer, draft=draft, mode='imaginative', permitted_bridges=[original])
    assert result.status == 'approved'
    assert original['skill'] == 'trend analysis'
    assert [b['skill'] for b in result.final_bridges] == ([expected] if expected else [])
    assert len(reviewer.calls) == (1 if choice == 'accept' else 2)
    if choice != 'accept':
        assert 'trend analysis' not in result.markdown


@pytest.mark.parametrize('replacement', ['expert analysis', 'data analysis\n- invented', 'data: inventory'])
def test_bad_narrowing_fails_closed(replacement):
    reviewer = Reviewer([bridge_answer('narrow', replacement)])
    result = run(reviewer, draft=SOURCE + '- trend analysis\n',
                 permitted_bridges=[valid_bridge()])
    assert result.status == 'withheld'


def test_duplicated_bridge_decision_never_approves():
    def duplicate(payload):
        out = bridge_answer('accept')(payload)
        out['bridge_decisions'].append(dict(out['bridge_decisions'][0]))
        return out
    result = run(Reviewer([duplicate]), draft=SOURCE + '- trend analysis\n',
                 permitted_bridges=[valid_bridge()])
    assert result.status == 'withheld'


def test_cannot_remove_a_sourced_skill_line():
    source = SOURCE + '- trend analysis\n'
    original = valid_bridge()
    reviewer = Reviewer([bridge_answer('remove')])
    result = asyncio.run(correct(source, {'document-1': source}, reviewer,
                                 permitted_bridges=[original]))
    assert result.status == 'withheld'


# --- Raw-versus-raw anchor validation against immutable original sources ---
#
# Bridge anchors are code-point offsets into the ORIGINAL candidate documents
# (that is where the Researcher computed them). The reviewer, however, receives
# redacted text. Redaction is length-preserving, so the offsets are identical in
# both views, but a redacted quote no longer equals the original slice. The deck
# must therefore compare the stored quote with the original text, not with its
# own re-redacted form; otherwise every bridge whose evidence contains a covered
# PII category is rejected and the whole analysis is withheld.

from disclosure import redact_sources  # noqa: E402

PRIVATE = 'person@example.test'
PII_SOURCE = (
    '## Professional Experience\n**Acme** -- City\n'
    f'- Compared weekly sales for {PRIVATE} to identify trends.\n'
    '## Projects\n## Education\n## Certifications\n## Technical Skills\n'
)
PII_QUOTE = f'Compared weekly sales for {PRIVATE} to identify trends.'


def pii_bridge():
    accepted, rejected = build_bridges(
        [{'skill': 'trend analysis', 'evidence_quote': PII_QUOTE,
          'source_id': 'document-1', 'rationale': 'General analytical capability.'}],
        sources={'document-1': PII_SOURCE}, mode='imaginative')
    assert len(accepted) == 1 and not rejected
    bridge = accepted[0].to_dict()
    assert PII_SOURCE[bridge['quote_start']:bridge['quote_end']] == PII_QUOTE
    assert PRIVATE in bridge['evidence_quote']
    return bridge


def redacted_view():
    """Return (redacted SourceDocuments, original pairs) mirroring the live route.

    ``app._run_source_correction`` reviews the redacted documents but passes the
    immutable originals as ``original_sources``; this helper keeps that exact
    split so the raw-versus-raw comparison is exercised, not a tautology.
    """
    redacted, attestation = redact_sources((('document-1', PII_SOURCE),))
    assert attestation is not None
    assert len(redacted[0][1]) == len(PII_SOURCE)
    assert PRIVATE not in redacted[0][1]
    originals = (('document-1', PII_SOURCE),)
    return tuple(SourceDocument(sid, text) for sid, text in redacted), originals


def run_pii(reviewer, bridges, *, originals=None, draft=None):
    sources, redacted = redacted_view()
    return asyncio.run(correct(
        draft if draft is not None else PII_SOURCE,
        sources, reviewer, mode='imaginative', permitted_bridges=bridges,
        original_sources=redacted if originals is None else originals))


def test_pii_quote_validates_raw_versus_raw_against_original_sources():
    bridge = pii_bridge()
    # The draft must carry the bridged skill line, otherwise the deck record
    # does not survive into the final document and is correctly dropped.
    draft = PII_SOURCE + '- trend analysis\n'
    reviewer = Reviewer([bridge_answer('accept')])
    result = run_pii(reviewer, [bridge], draft=draft)
    assert result.status == 'approved'
    assert [b['skill'] for b in result.final_bridges] == ['trend analysis']
    assert result.markdown == draft
    assert len(reviewer.calls) == 1
    # The reviewer still saw redacted evidence, never the original address.
    serialized = json.dumps(reviewer.calls)
    assert PRIVATE not in serialized
    assert '[email removed]' in serialized


@pytest.mark.parametrize('mutation', ['altered_quote', 'wrong_source_id', 'start_zero_end_zero',
                                      'end_past_document', 'negative_start', 'start_after_end'])
def test_pii_bridge_anchor_rejections_are_unchanged(mutation):
    bridge = pii_bridge()
    if mutation == 'altered_quote':
        bridge['evidence_quote'] = bridge['evidence_quote'].replace(
            PRIVATE, 'otherx@example.test')
    elif mutation == 'wrong_source_id':
        bridge['source_id'] = 'other-document'
    elif mutation == 'start_zero_end_zero':
        bridge['quote_start'] = bridge['quote_end'] = 0
    elif mutation == 'end_past_document':
        bridge['quote_end'] = len(PII_SOURCE) + 1
    elif mutation == 'negative_start':
        bridge['quote_start'] = -1
    elif mutation == 'start_after_end':
        bridge['quote_start'], bridge['quote_end'] = (bridge['quote_end'],
                                                      bridge['quote_start'])
    reviewer = Reviewer([bridge_answer('accept')])
    result = run_pii(reviewer, [bridge])
    assert result.status == 'withheld' and result.reason == 'unresolved_claims'
    assert result.markdown is None
    assert not reviewer.calls


def test_pii_bridge_without_originals_fails_closed():
    """Omitted original_sources falls back to the redacted snapshot, so a PII
    anchor cannot be proven and the review is withheld rather than guessed."""
    sources, _ = redacted_view()
    bridge = pii_bridge()
    reviewer = Reviewer([bridge_answer('accept')])
    result = asyncio.run(correct(PII_SOURCE, sources, reviewer, mode='imaginative',
                                 permitted_bridges=[bridge]))
    assert result.status == 'withheld' and result.reason == 'unresolved_claims'
    assert not reviewer.calls
