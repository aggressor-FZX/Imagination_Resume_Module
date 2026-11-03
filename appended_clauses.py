"""Tier-B work list: appended clauses that the cited evidence does not attest.

Why this module exists
----------------------

Run 10 (`850ee3ea-ca2e-4210-92b4-0e57ed5149eb`, revision `47c4254`) was approved on
its first review with ``repair_calls = 0`` and the publication gate returned ``[]``.
It still carried six bullets that append an outcome the source never states::

    Installed and maintained commercial electrical systems across 40+ job sites,
    ensuring compliance with NFPA 70E standards.

Two separate failures combined to let that through:

1. ``_introduces_beyond`` inspects the span for **capitalised names** and
   **unbacked figures**. ``ensuring``, ``compliance``, ``workforce``, ``reliability``
   are lowercase words, so the check returns ``[]`` and no repair is triggered.
2. The reviewer answered the **line-level** indexed contract, marking whole lines
   ``supported`` without naming evidence for each clause inside them. The engine is
   therefore told "this line is established" and has nothing finer-grained to test.

A vocabulary rule cannot stand in for semantics
----------------------------------------------

The obvious "fix" — treat words absent from the source as fabrication — is wrong in
both directions, and the run-10 data proves it:

* ``gaining hands-on electrical construction experience`` is a *reasonable
  paraphrase* of supervised electrical field work. It introduces only ``gain`` and
  ``hand``; the activity it characterises is exactly what the source bullet states.
  A word-list rule would delete a legitimate sentence.
* ``ensuring continuous operations`` introduces only ``ensuring``, ``continuous``
  and ``operations`` — the same *shape* — but asserts an operational-continuity
  guarantee that no source describes.

Measured on the six live clauses, a "no content word appears anywhere in the corpus"
rule fires on **one of six**; a "no content word appears in the evidence cited for
this line" rule fires on **five of five** it was given. Neither number is a verdict
about the candidate's writing. They are measurements of how much the deterministic
layer can see.

What this module therefore does
-------------------------------

It does **not** decide fabrication and it does **not** gate publication. It produces
a *work list*: for each supported line, the comma-delimited clauses that share no
content word with the evidence cited for that line. That is exactly the set a
semantic reviewer must justify, per SPEC.v2's separation of deterministic Tier-A
checks from semantic Tier-B review.

Two consequences are deliberate:

* a faithful paraphrase whose every word is attested produces an empty list, so the
  check costs a correct draft nothing;
* a clause that *is* a legitimate inference still appears on the list. That is not a
  false positive to be suppressed; it is the question Tier B has to answer, and
  answering it in the engine would be exactly the "mechanical check passed,
  therefore it is supported" error SPEC.v2 warns about.

Labelled **[tier-b-worklist]**.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from source_correction import _content_words, _stem

# A clause boundary that introduces a trailing participial phrase:
#
#   "… across 40+ job sites, ensuring compliance …"
#   "… for 12 commercial buildings, ensuring continuous operations."
#
# Only a LOWERCASE ``-ing`` word after the comma counts. A capitalised token starts a
# new proper-noun phrase (", NFPA 70E"), and "…, and …" continues a list. This is a
# boundary detector, not a judgement: it locates the clause, it never rules on it.
_APPENDED_CLAUSE = re.compile(r",\s*(?=(?:[a-z][a-z'\u2019-]*ing)\b)")


def appended_clauses(line: str) -> list[tuple[int, int, str]]:
    """Locate trailing participial clauses. Returns ``(start, end, text)`` triples.

    Offsets index into ``line`` so a caller can cite the clause to a reviewer
    without re-deriving it. The returned text excludes the leading comma and any
    trailing full stop, because neither is part of the assertion.
    """
    found: list[tuple[int, int, str]] = []
    for match in _APPENDED_CLAUSE.finditer(line or ""):
        start = match.end()
        tail = line[start:]
        # Stop at a sentence end; a second sentence is not an appended clause.
        stop = tail.find(". ")
        end = start + (stop if stop != -1 else len(tail))
        text = line[start:end].strip().rstrip(".")
        if text:
            found.append((start, start + len(text), text))
    return found


def _content_stems(text: str) -> set[str]:
    return {_stem(word) for word in _content_words(text)}


def unattested_clauses(line: str, cited_evidence: Iterable[str]) -> list[dict[str, Any]]:
    """Appended clauses with no content-word support in ``cited_evidence``.

    ``cited_evidence`` must be the evidence the reviewer named **for this line**,
    not the whole document it came from. Using the whole document is what hid the
    defect in run 10: ``compliance`` and ``standard`` both occur somewhere in
    ``document-0`` (on the training bullet and the certification line), so a
    document-scoped comparison reported the installation bullet as fully attested
    while a line-scoped comparison reports the appended clause as unbacked.

    A clause is reported when it introduces at least one content word and shares
    **none** of them with the cited evidence. Partial support is not reported: a
    clause that restates a sourced fact in rewritten words is the paraphrase the
    system exists to permit, and a threshold would turn this into the vocabulary
    filter the module docstring rejects.

    Each finding carries the clause text and its offsets. It carries no verdict.
    """
    evidence_stems = _content_stems(" ".join(cited_evidence))
    findings: list[dict[str, Any]] = []
    for start, end, text in appended_clauses(line):
        clause_stems = _content_stems(text)
        if not clause_stems:
            continue
        if clause_stems & evidence_stems:
            continue
        findings.append({
            "start": start,
            "end": end,
            "clause": text,
            # Recorded so a Tier-B reader can see what the clause contributes;
            # it is not itself a reason to reject.
            "unattested": sorted(clause_stems - evidence_stems),
        })
    return findings


def worklist(lines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the Tier-B work list from ``[{line_id, text, evidence}]`` entries.

    ``evidence`` is the exact source text the reviewer cited for that line. A line
    with no cited evidence is skipped: the engine already withholds a supported
    line that names no evidence, so there is nothing here to add.
    """
    items: list[dict[str, Any]] = []
    for entry in lines:
        evidence = entry.get("evidence") or []
        if not evidence:
            continue
        found = unattested_clauses(entry.get("text") or "", evidence)
        if found:
            items.append({"line_id": entry["line_id"], "clauses": found})
    return items
