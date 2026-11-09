import asyncio
import json
from pathlib import Path

import pytest

import decision_contract as dc
from experiment.binding_candidate import run_binding, safe_repair
from experiment.binding_audit_packet import build, verify
from experiment.cases_v2 import SOURCES, by_id
from experiment.replay_binding import replay

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / 'results/comparison-v2b-20261109/attempts.json'


def test_all_28_attempts_and_five_release_overrides():
    rows = replay(json.loads(RETAINED.read_text()))
    assert len(rows) == 28
    assert {(r['cell'], r['case_id']) for r in rows if r['release_policy_override']} == {
        ('revised+advisory', 'faithful_paraphrase'),
        ('revised+advisory', 'supported_method'),
        ('revised+advisory', 'wrong_record_attribution'),
        ('revised+advisory', 'transferable_skill'),
        ('revised+advisory', 'omitted_record'),
    }
    assert all(not row['rule_changed_draft'] for row in rows)
    assert sum(bool(r['re_review_present']) for r in rows) == 7
    assert all(r['re_review_raw_model_decision'] == 'revise' for r in rows if r['re_review_present'])
    truncated = next(r for r in rows if r['error'] and 'TruncatedResponse' in r['error'])
    assert truncated['first_review_raw'] is None and truncated['finish_and_usage'] is None


def test_audit_packet_preserves_all_retained_fields(tmp_path):
    attempts = json.loads(RETAINED.read_text())
    target = tmp_path / 'audit'
    build(attempts, target)
    verify(attempts, target)
    entries = json.loads((target / 'blind/entries.json').read_text())
    assert len(entries) == 28
    assert sum(e['first_review'] is None for e in entries) == 1
    assert all('cell' not in e and 'case_id' not in e for e in entries)
    assert (target / 'private-key.json').is_file()
    assert not (target / 'blind/private-key.json').exists()


APPROVE = 'Decision: approve\n\nIssues:\n- none\n'
REVISE = ('Decision: revise\n\nIssues:\n'
          '- passage: Cut rework by 18% through targeted quality audits.\n'
          '  problem: unsupported\n'
          '  sources: [candidate-main.txt]\n'
          '  why: unsupported method\n'
          '  correction: Cut rework by 18% by standardizing panel labeling across the terminal fleet.\n')


def run(draft, answers, rules=None):
    seen = []
    async def ask(current, sources):
        seen.append(current)
        return answers[len(seen) - 1]
    return asyncio.run(run_binding(draft, SOURCES, ask=ask, observe_rules=rules)), seen


def test_faithful_approval_is_binding_heuristics_observe_only():
    draft = by_id('faithful_paraphrase').draft_before_trim
    result, calls = run(draft, [APPROVE], lambda _d, _s: ['heuristic would trim'])
    assert result.release and result.release_reason == 'initial_llm_approve'
    assert result.rule_observations == ['heuristic would trim']
    assert calls == [draft] and result.proposed_document == draft


def test_all_retained_revise_releases_do_not_release_on_first_review():
    attempts = json.loads(RETAINED.read_text())
    selected = [a for a in attempts if a['cell'] == 'revised+advisory' and a['outcome'] == 'approved']
    assert len(selected) == 5
    for a in selected:
        # A repeated revise suffices as any re-review: it may never authorize release.
        result, _ = run(a['draft_after_trim'], [a['provider_response'], 'Decision: revise\n'])
        assert result.initial_decision == 'revise'
        assert result.release is False, a['case_id']


@pytest.mark.parametrize('second,allowed', [(APPROVE, True), ('Decision: revise\n', False),
                                           ('Decision: cannot_assess\n', False), ('not a decision', False)])
def test_correction_requires_new_valid_approval(second, allowed):
    draft = ('## Professional Experience\n**Cascade Freight Partners** -- Kent, WA\n'
             '*Cost Reporting Coordinator | 07/2015 - 03/2019*\n'
             '- Cut rework by 18% through targeted quality audits.\n')
    result, calls = run(draft, [REVISE, second])
    assert result.repair_status == 'proposed_exact_owned_source_bullet'
    assert len(calls) == 2 and calls[1] == result.proposed_document and calls[1] != draft
    assert result.release is allowed


def test_safe_repair_rejects_duplicate_omission_commentary_and_wrong_owner():
    draft = ('## Professional Experience\n**Cascade Freight Partners** -- Kent, WA\n'
             '- Cut rework by 18% through targeted quality audits.\n')
    issue = dc.Issue('Cut rework by 18% through targeted quality audits.', 'unsupported', '',
                     correction='Cut rework by 18% by standardizing panel labeling across the terminal fleet.')
    assert safe_repair(draft + '- Cut rework by 18% through targeted quality audits.\n', SOURCES, [issue])[0] != ''
    assert safe_repair(draft + '- Cut rework by 18% through targeted quality audits.\n', SOURCES, [issue])[1] == 'passage_not_unique_literal_bullet'
    for replacement in ['*(omit)*', '"(omit)" (Note: check)', issue.correction + ' Approved as corrected.']:
        issue.correction = replacement
        assert safe_repair(draft, SOURCES, [issue])[0] == draft
    issue.problem = 'omitted'
    issue.correction = '- Cut rework by 18% by standardizing panel labeling across the terminal fleet.'
    assert safe_repair(draft, SOURCES, [issue])[0] == draft
    issue.problem = 'unsupported'
    issue.correction = 'Reconciled 1,180 fuel invoices against depot dispensing records each quarter.'
    assert safe_repair(draft, SOURCES, [issue])[1] == 'replacement_not_exact_owned_source_bullet'


def test_all_offered_references_must_resolve():
    issue = dc.Issue('passage', 'unsupported', '', sources=['candidate-main.txt', 'missing.txt'])
    decision = dc.Decision('revise', issues=[issue])
    from experiment.binding_candidate import references_resolve
    assert not references_resolve(decision, SOURCES)
    issue.sources = ['candidate-main.txt', 'candidate-supplement.txt']
    assert references_resolve(decision, SOURCES)