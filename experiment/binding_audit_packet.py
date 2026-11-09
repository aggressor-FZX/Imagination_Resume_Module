"""Build a blinded, complete packet from the retained matrix without network access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from experiment.cases_v2 import SOURCES


def build(attempts: list[dict], out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    blind = out / 'blind'
    blind.mkdir(parents=True)
    (blind / 'sources.json').write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2) + '\n')
    order = list(range(len(attempts)))
    random.Random(20261109).shuffle(order)
    private = {}
    entries = []
    for number, index in enumerate(order, 1):
        a = attempts[index]
        label = f'AUDIT-{number:02d}'
        entry = {
            'label': label,
            'original_draft': a.get('draft_before_trim') or None,
            'reviewed_draft': a.get('draft_after_trim') or None,
            'first_review': a.get('provider_response') or None,
            're_review': a.get('recheck_response') or None,
            'final_document': a.get('final_markdown') or None,
            'missing_data_note': ('Review or final document unavailable; do not reconstruct it.'
                                  if not a.get('provider_response') or not a.get('final_markdown') else None),
        }
        entries.append(entry)
        private[label] = {'index': index, 'cell': a['cell'], 'case_id': a['case_id']}
    (blind / 'entries.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n')
    (blind / 'INSTRUCTIONS.md').write_text(
        '# Blind audit\n\nJudge each entry using the exact sources. For each review assess whether its '
        'substantive objections are supported, wrong, or unclear, with source and owner. Judge both '
        'reviewed and final documents separately for unsupported facts, misattribution, omission, '
        'and injected reviewer commentary; explain whether each is deliverable. Do not infer '
        'correctness from formatting, references, withhold, or initial expectations. Mark absent '
        'text unassessable. Do not access the sibling private-key.json. Raw response format may '
        'reveal the contract; it cannot be fully blinded without altering evidence.\n')
    (out / 'private-key.json').write_text(json.dumps(private, indent=2) + '\n')


def verify(attempts: list[dict], out: Path) -> None:
    blind = out / 'blind'
    assert json.loads((blind / 'sources.json').read_text()) == SOURCES
    entries = json.loads((blind / 'entries.json').read_text())
    key = json.loads((out / 'private-key.json').read_text())
    assert len(entries) == len(attempts) == len(key)
    assert sorted(v['index'] for v in key.values()) == list(range(len(attempts)))
    fields = {'original_draft': 'draft_before_trim', 'reviewed_draft': 'draft_after_trim',
              'first_review': 'provider_response', 're_review': 'recheck_response',
              'final_document': 'final_markdown'}
    for entry in entries:
        a = attempts[key[entry['label']]['index']]
        assert all(entry[field] == (a.get(original) or None) for field, original in fields.items())
        assert not ({'cell', 'case_id', 'outcome', 'expectation'} & entry.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=Path, default=Path('results/comparison-v2b-20261109/attempts.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.attempts.read_bytes()
    attempts = json.loads(raw)
    build(attempts, args.out)
    verify(attempts, args.out)
    print(f'Verified {len(attempts)} entries; retained input sha256={hashlib.sha256(raw).hexdigest()}')


if __name__ == '__main__':
    main()