"""Offline exercises of the JSON engine and its real adapter boundary."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from semantic_review.json_engine import correct
from source_correction import SourceDocument


TAIL = "## Projects\n## Education\n## Certifications\n## Technical Skills\n"
SOURCE = ("## Professional Experience\n"
          "**Acme** -- City\n- Managed inventory.\n- Tracked orders.\n"
          "**Beta** -- City\n- Processed invoices.\n" + TAIL)
DRAFT = ("## Professional Experience\n"
         "**Acme** -- City\n- Managed inventory. using a fictional tool\n"
         "**Beta** -- City\n- Processed invoices. using fictional audits\n" + TAIL)


def answer(decision='approve', issues=None, rationale='private model rationale'):
    return {'version': 'semantic_review_v1', 'decision': decision,
            'rationale': rationale, 'issues': issues or []}


def issue(passage, source, correction, problem='unsupported'):
    return {'operation': 'replace', 'passage': passage, 'problem': problem, 'sources': [source],
            'why': 'private explanation', 'correction': correction}


def insert(destination, source):
    return {'operation': 'insert_record', 'destination': destination, 'problem': 'omitted',
            'sources': [source], 'why': 'missing record'}


def delete(passage, source):
    return {'operation': 'delete', 'passage': passage, 'problem': 'unsupported',
            'sources': [source], 'why': 'unsupported bullet'}


def views(payload):
    return json.loads(payload['text']), [json.loads(s['text']) for s in payload['sources']]


class Reviewer:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.budgets = []

    def set_budget(self, seconds):
        self.budgets.append(seconds)

    async def __call__(self, phase, payload):
        self.calls.append((phase, copy.deepcopy(payload)))
        response = self.responses.pop(0)
        if callable(response):
            response = response(payload)
        if isinstance(response, Exception):
            raise response
        return response


def run(reviewer, draft=DRAFT, sources=None, **kwargs):
    return asyncio.run(correct(draft, sources if sources is not None else {'private-name': SOURCE},
                               reviewer, **kwargs))


def replacement(payload, index=0, source_index=0, correction='Managed inventory.'):
    doc, sources = views(payload)
    return answer('revise', [issue(doc['records'][index]['passages'][0]['id'],
                                   sources[0]['records'][source_index]['id'], correction)])


def test_clean_approval_preserves_document_and_no_rationale():
    r = Reviewer(answer())
    result = run(r, draft=SOURCE)
    assert result.status == 'approved' and result.markdown == SOURCE
    assert result.commentary == '' and result.coverage.level == 'semantic_review_not_proof'
    assert result.coverage.claim_count == result.coverage.supported_count == 0
    assert result.review_calls == 1 and result.repair_calls == 0
    assert r.calls[0][0] == 'semantic_review'
    assert set(r.calls[0][1]) == {'contract', 'sources', 'text'}
    assert r.calls[0][1]['sources'][0]['source_id'] != 'private-name'


def test_source_snapshot_and_full_fresh_rereview():
    sources = {'private-name': SOURCE}
    def first(payload):
        sources['private-name'] = 'changed after call'
        return replacement(payload)
    r = Reviewer(first, answer())
    result = run(r, sources=sources)
    assert result.status == 'approved' and result.repair_calls == 1
    assert 'fictional tool' not in result.markdown
    assert 'fictional audits' in result.markdown
    assert views(r.calls[1][1])[0]['document'] == result.markdown
    assert views(r.calls[1][1])[1][0]['document'] == SOURCE
    assert result.review_calls == 2 and r.budgets[1] <= r.budgets[0]


def test_multi_record_batch_and_conflicting_targets():
    def batch(payload):
        doc, src = views(payload)
        return answer('revise', [issue(doc['records'][0]['passages'][0]['id'], src[0]['records'][0]['id'], 'Managed inventory.'),
                                 issue(doc['records'][1]['passages'][0]['id'], src[0]['records'][1]['id'], 'Processed invoices.')])
    result = run(Reviewer(batch, answer()))
    assert result.status == 'approved' and 'fictional' not in result.markdown
    def conflict(payload):
        response = replacement(payload)
        response['issues'].append(copy.deepcopy(response['issues'][0]))
        return response
    r = Reviewer(conflict)
    assert run(r).status == 'withheld' and len(r.calls) == 1


def test_multiple_nonoverlapping_edits_in_one_record():
    draft = DRAFT.replace('**Beta**', '- Tracked orders. using a fictional method\n**Beta**')
    def batch(payload):
        doc, src = views(payload)
        passages = doc['records'][0]['passages']
        sid = src[0]['records'][0]['id']
        return answer('revise', [issue(passages[0]['id'], sid, 'Managed inventory.'),
                                 issue(passages[1]['id'], sid, 'Tracked orders.')])
    result = run(Reviewer(batch, answer()), draft=draft)
    assert result.status == 'approved'
    assert '- Managed inventory.\n- Tracked orders.\n' in result.markdown


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('problem', ['unsupported', 'misattributed'])
def test_delete_refuses_even_apparently_unsupported_material(index, problem):
    draft = DRAFT.replace('**Beta**', '- Imaginary claim.\n**Beta**')
    def select(payload):
        doc, src = views(payload)
        item = delete(doc['records'][0]['passages'][index]['id'], src[0]['records'][0]['id'])
        item['problem'] = problem
        return answer('revise', [item])
    reviewer = Reviewer(select, answer())
    result = run(reviewer, draft=draft)
    assert result.status == 'withheld' and result.reason == 'unresolved_claims'
    assert result.markdown is None and result.repair_calls == 0 and len(reviewer.calls) == 1


@pytest.mark.parametrize('paraphrase', [
    'Oversaw stock.',
    'Oversaw stock and invented an automated warehouse.',
    'Supervised the flow of goods.',
])
def test_delete_refuses_supported_paraphrases_without_literal_overlap(paraphrase):
    draft = SOURCE.replace('Managed inventory.', paraphrase)
    def select(payload):
        doc, src = views(payload)
        return answer('revise', [delete(doc['records'][0]['passages'][0]['id'],
                                        src[0]['records'][0]['id'])])
    assert 'Managed inventory.' not in paraphrase and paraphrase not in SOURCE
    reviewer = Reviewer(select, answer())
    result = run(reviewer, draft=draft)
    assert result.reason == 'unresolved_claims' and result.markdown is None
    assert result.repair_calls == 0 and len(reviewer.calls) == 1


@pytest.mark.parametrize('extracted_first', [True, False])
def test_delete_refuses_fact_from_unindexed_extracted_document_in_either_order(extracted_first):
    fixture_path = Path(__file__).parent / 'fixtures/tier_b/extracted_format_excerpt.json'
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    extracted = fixture['sources']['document-1']['text']
    assert 'Reduced after-hours callouts by 27% by adding thermographic inspections.' in extracted
    # A valid indexed record makes the delete structurally actionable. The related
    # fact is in a different, bare DOCX extraction that has no indexed records.
    marked = '## Professional Experience\n**Acme** -- City\n- Managed inventory.\n' + TAIL
    draft = marked.replace('Managed inventory.',
                           'Cut emergency calls by 27% through thermal checks.')
    sources = ({'extracted': extracted, 'marked': marked} if extracted_first
               else {'marked': marked, 'extracted': extracted})
    def select(payload):
        doc, src = views(payload)
        indexed = [record for document in src for record in document['records']
                   if record['owner'] == 'Acme']
        assert len(indexed) == 1
        assert any('Riverbend Facilities Group' in document['document'] and not document['records']
                   for document in src)
        return answer('revise', [delete(doc['records'][0]['passages'][0]['id'], indexed[0]['id'])])
    reviewer = Reviewer(select, answer())
    result = run(reviewer, draft=draft, sources=sources)
    assert result.reason == 'unresolved_claims' and result.markdown is None
    assert result.repair_calls == 0 and len(reviewer.calls) == 1


@pytest.mark.parametrize('delete_first', [True, False])
def test_delete_rejects_entire_mixed_edit_batch(delete_first):
    draft = DRAFT.replace('**Beta**', '- Oversaw stock.\n**Beta**')
    def batch(payload):
        doc, src = views(payload)
        source_id = src[0]['records'][0]['id']
        items = [issue(doc['records'][0]['passages'][0]['id'], source_id, 'Managed inventory.'),
                 delete(doc['records'][0]['passages'][1]['id'], source_id)]
        return answer('revise', list(reversed(items)) if delete_first else items)
    reviewer = Reviewer(batch, answer())
    result = run(reviewer, draft=draft)
    assert result.reason == 'unresolved_claims' and result.markdown is None
    assert result.repair_calls == 0 and len(reviewer.calls) == 1


def test_replace_cannot_discard_literal_bullet_from_another_owner():
    draft = DRAFT.replace('using a fictional tool', 'Processed invoices.')
    assert run(Reviewer(replacement), draft=draft).status == 'withheld'


def test_insertion_rejects_multiple_records_at_same_section_boundary():
    source = SOURCE.replace('## Projects', '**Gamma** -- City\n- Filed forms.\n## Projects')
    draft = DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '')
    def batch(payload):
        doc, src = views(payload)
        dest = doc['sections'][0]['id']
        return answer('revise', [insert(dest, src[0]['records'][1]['id']),
                                 insert(dest, src[0]['records'][2]['id'])])
    assert run(Reviewer(batch), draft=draft, sources={'one': source}).status == 'withheld'


def test_italic_header_cannot_launder_other_record():
    source = SOURCE.replace('- Managed inventory.', '*Other Employer*\n- Managed inventory.')
    assert run(Reviewer(replacement), sources={'one': source}).status == 'withheld'
    draft = DRAFT.replace('- Managed inventory.', '*Other Employer*\n- Managed inventory.')
    assert run(Reviewer(replacement), draft=draft).status == 'withheld'


@pytest.mark.parametrize('missing', ['Professional Experience', 'Projects', 'Education',
                                    'Certifications', 'Technical Skills'])
def test_approval_requires_each_core_heading(missing):
    draft = SOURCE.replace('## ' + missing + '\n', '')
    assert run(Reviewer(answer()), draft=draft).reason == 'source_section_missing'


def test_bare_and_alias_section_headings_use_existing_conventions():
    draft = SOURCE.replace('## Professional Experience', 'Work Experience').replace(
        '## Technical Skills', 'Skills')
    assert run(Reviewer(answer()), draft=draft).status == 'approved'


@pytest.mark.parametrize('mutate', [
    lambda i: i.pop('operation'),
    lambda i: i.update(operation='delete'),
    lambda i: i.update(operation='insert_record'),
    lambda i: i.pop('correction'),
    lambda i: i.update(operation='unknown'),
    lambda i: i.update(sources=[]),
])
def test_operation_shape_is_strict(mutate):
    def malformed(payload):
        result = replacement(payload)
        mutate(result['issues'][0])
        return result
    assert run(Reviewer(malformed, malformed)).reason == 'malformed_output'


def test_record_insertion_rejects_duplicate_wrong_section_and_protected_source():
    def insertion(destination=0, source=1):
        def build(payload):
            doc, src = views(payload)
            return answer('revise', [insert(doc['sections'][destination]['id'],
                                            src[0]['records'][source]['id'])])
        return build
    assert run(Reviewer(insertion())).status == 'withheld'  # already exists
    draft = DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '')
    assert run(Reviewer(insertion(1)), draft=draft).status == 'withheld'
    pii = SOURCE.replace('Processed invoices.', 'Processed invoices. a@example.test')
    r = Reviewer(insertion())
    assert run(r, draft=draft, sources={'one': pii}).status == 'withheld'
    assert 'a@example.test' not in json.dumps(r.calls)


def test_explicit_destination_inserts_whole_record():
    draft = DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '')
    def missing(payload):
        doc, src = views(payload)
        return answer('revise', [insert(doc['sections'][0]['id'], src[0]['records'][1]['id'])])
    result = run(Reviewer(missing, answer()), draft=draft)
    assert result.status == 'approved' and result.restore_calls == 1
    assert '**Beta** -- City\n- Processed invoices.\n' in result.markdown
    assert result.markdown.index('**Beta**') < result.markdown.index('## Projects')


def test_whole_record_keeps_all_bullets_and_crlf():
    source = SOURCE.replace('- Processed invoices.\n', '- Processed invoices.\n- Filed receipts.\n').replace('\n', '\r\n')
    draft = DRAFT.replace('**Beta** -- City\n- Processed invoices. using fictional audits\n', '').replace('\n', '\r\n')
    def missing(payload):
        doc, src = views(payload)
        return answer('revise', [insert(doc['sections'][0]['id'], src[0]['records'][1]['id'])])
    result = run(Reviewer(missing, answer()), draft=draft, sources={'one': source})
    assert result.status == 'approved'
    assert '**Beta** -- City\r\n- Processed invoices.\r\n- Filed receipts.' in result.markdown


@pytest.mark.parametrize('mutation', ['foreign_source', 'caller_source', 'bad_passage', 'wrong_owner',
                                      'invented', 'no_change', 'bad_destination', 'unresolved'])
def test_untrusted_or_unsafe_edits_withhold(mutation):
    def build(payload):
        doc, src = views(payload)
        i = issue(doc['records'][0]['passages'][0]['id'], src[0]['records'][0]['id'], 'Managed inventory.')
        if mutation == 'foreign_source': i['sources'] = [payload['sources'][0]['source_id']]
        if mutation == 'caller_source': i['sources'] = ['private-name']
        if mutation == 'bad_passage': i['passage'] = 'not-an-id'
        if mutation == 'wrong_owner': i['sources'] = [src[0]['records'][1]['id']]
        if mutation == 'invented': i['correction'] = 'Invented tool.'
        if mutation == 'no_change': i['correction'] = 'Managed inventory. using a fictional tool'
        if mutation == 'bad_destination': i = insert(doc['records'][0]['id'], src[0]['records'][0]['id'])
        if mutation == 'unresolved': i['problem'] = 'unresolved'
        return answer('revise', [i])
    r = Reviewer(build)
    assert run(r).status == 'withheld'
    assert len(r.calls) == (2 if mutation == 'unresolved' else 1)


def test_duplicate_owner_and_wrapped_bullet_refuse():
    duplicate = SOURCE.replace('## Projects', '**Acme** -- Another city\n- Other work.\n## Projects')
    assert run(Reviewer(replacement), sources={'one': duplicate}).status == 'withheld'
    wrapped = DRAFT.replace(' using a fictional tool\n', '\n  using a fictional tool\n')
    def fake(payload):
        doc, src = views(payload)
        assert not doc['records'][0]['passages']
        return answer('revise', [issue('made-up', src[0]['records'][0]['id'], 'Managed inventory.')])
    assert run(Reviewer(fake), draft=wrapped).status == 'withheld'


def test_owner_is_not_enough_when_record_role_or_header_differs():
    different_role = DRAFT.replace('**Acme** -- City\n', '**Acme** -- City\n*Manager*\n')
    assert run(Reviewer(replacement), draft=different_role).status == 'withheld'
    different_location = DRAFT.replace('**Acme** -- City', '**Acme** -- Elsewhere')
    assert run(Reviewer(replacement), draft=different_location).status == 'withheld'


def test_insertion_refuses_ambiguous_record_end():
    draft = DRAFT.replace('**Beta**', 'unclassified trailing content\n**Beta**')
    def missing(payload):
        doc, src = views(payload)
        return answer('revise', [insert(doc['sections'][0]['id'], src[0]['records'][1]['id'])])
    assert run(Reviewer(missing), draft=draft).status == 'withheld'


def test_unrecognized_header_cannot_launder_another_employers_bullet():
    source = ('## Professional Experience\n**Acme** -- City\n- Inventory.\n'
              'Other Employer\n- Managed inventory.\n')
    assert run(Reviewer(replacement), sources={'one': source}).status == 'withheld'


def test_protected_target_and_protected_source_refuse_but_unrelated_pii_survives():
    pii = DRAFT.replace('Managed inventory. using a fictional tool',
                        'Managed inventory. contact a@example.test')
    r = Reviewer(replacement)
    assert run(r, draft=pii).status == 'withheld'
    assert 'a@example.test' not in json.dumps(r.calls)
    source_pii = SOURCE.replace('Managed inventory.', 'Managed inventory. a@example.test')
    assert run(Reviewer(replacement), sources={'a': source_pii}).status == 'withheld'
    unrelated = 'Jane | a@example.test\n' + DRAFT
    r = Reviewer(replacement, answer())
    result = run(r, draft=unrelated)
    assert result.status == 'approved' and result.markdown.startswith('Jane | a@example.test')
    assert 'a@example.test' not in json.dumps(r.calls)


def test_format_retry_is_once_and_never_contains_verdict_hint():
    r = Reviewer(ValueError('secret'), answer())
    result = run(r, draft=SOURCE)
    assert result.status == 'approved' and result.review_calls == 2
    a, b = r.calls[0][1], r.calls[1][1]
    assert b['text'] == a['text'] and b['sources'] == a['sources']
    assert 'invalid JSON shape' in b['contract'] and 'private model rationale' not in b['contract']
    assert run(Reviewer({}, {})).reason == 'malformed_output'


@pytest.mark.parametrize('decision', ['cannot_assess', 'revise'])
def test_no_approval_from_refusal_or_second_revision(decision):
    result = run(Reviewer(replacement, answer(decision)))
    assert result.status == 'withheld' and result.markdown is None


def test_marker_and_section_contract_fail_closed():
    assert run(Reviewer(answer()), draft=SOURCE + '- (omitted)\n').status == 'withheld'
    assert run(Reviewer(answer()), draft='## Professional Experience\n').reason == 'source_section_missing'


def test_mode_and_bridges():
    r = Reviewer(answer())
    assert run(r, draft=SOURCE, mode='imaginative').status == 'approved'
    assert 'IMAGINATIVE' in r.calls[0][1]['contract']
    r = Reviewer(answer())
    assert run(r, permitted_bridges=[{'skill': 'reporting'}], mode='imaginative').status == 'withheld'
    assert not r.calls


def test_deadline_cancels_slow_provider():
    async def slow(_phase, _payload):
        await asyncio.Event().wait()
    result = run(slow, timeout_seconds=0.01)
    assert result.reason == 'timeout_initial_review' and result.markdown is None


def test_real_adapter_boundary_with_in_memory_transport(monkeypatch):
    import correction_provider as cp
    monkeypatch.setenv(cp.DISCLOSURE_SETTING, '1')
    class Response:
        status = 200
        headers = {}
        content = None
        def __init__(self): self.content = self
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def iter_chunked(self, size):
            yield json.dumps({'model': 'offline/model', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
                              'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer())}}]}).encode()
    class Session:
        def post(self, *args, **kwargs): return Response()
    result = run(cp.CorrectionProvider('fake', Session(), model='offline/model'), draft=SOURCE,
                 sources=(SourceDocument('s', SOURCE),))
    assert result.status == 'approved'


@pytest.mark.parametrize('operation', ['replace', 'delete', 'insert_record'])
def test_real_adapter_accepts_current_revision_schema(operation):
    import correction_provider as cp
    new_issue = {'replace': lambda: issue('pabc', 'rabc', 'Managed inventory.'),
                 'delete': lambda: delete('pabc', 'rabc'),
                 'insert_record': lambda: insert('sabc', 'rabc')}[operation]()
    assert cp._validate_semantic_response(answer('revise', [new_issue])) is None