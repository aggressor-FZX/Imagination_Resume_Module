"""Scoped repair: apply a reviewer's correction without widening what may change.

Production copy of `experiment/repair.py`. The experiment retains its own copy so
past runs stay reproducible against the code that produced them; this copy is the
one the live service imports. Any behavioural change must be made in BOTH, and a
test asserts they have not diverged.

What this replaces, and why
---------------------------
`safe_repair` accepted a correction only when it was a BYTE-EXACT source bullet. That
refused Gemma's correct correction for `unsupported_method`:

    draft:      Cut rework by 18% through standardizing panel labeling and targeted
                quality audits across the terminal fleet.
    correction: Cut rework by 18% through standardizing panel labeling across the
                terminal fleet.
    source:     Cut rework by 18% by standardizing panel labeling across the
                terminal fleet.

The model wrote "through" where the source writes "by". The clause it removed is the
invented one, and every surviving token is supported. Requiring byte-identity rejects
ordinary paraphrase, which the contract explicitly permits — so correct repairs could
never be applied and no corrected document could ever be released.

The four gaps this module closes
--------------------------------
1. faithful paraphrase accepted (not byte-identical to the source)
2. the edit is located inside the CORRECT owner record, unambiguously
3. a missing source-backed record can be restored at a defined destination
4. unsupported additions are removed while sourced facts survive

And the four it must keep closed
--------------------------------
- ambiguous target refused, never first-match
- invented dates/metrics refused
- employers never merged
- qualifications never moved between records

How paraphrase is permitted without permitting invention
--------------------------------------------------------
The correction is compared to the source bullets FOR THE SAME OWNER as a multiset of
CONTENT TOKENS after normalisation. Differences in a small set of connective words
("by"/"through", "cut"/"reduced", "with"/"against", "tracked"/"monitored", "built"/
"produced", "sent"/"reported", "every"/"all") are ignored; no other token may be
added. So "through" for "by" passes and a NEW method, metric or date cannot.

That is deliberately not a general-purpose editor. It only ever rewrites ONE existing
bullet in place, or appends ONE record whose every line comes from the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import decision_contract as dc

_HEAD = re.compile(r'^\*\*(.+?)\*\*\s*--\s*(.*)$')
_ROLE = re.compile(r'^\*(.+?)\*$')

# Words whose substitution does not change what the document asserts. This is a
# CLOSED list; nothing outside it may differ between a correction and the source.
_CONNECTIVES = {
    "by", "through", "via", "with", "against", "in", "during", "on", "at", "for",
    "every", "each", "all", "the", "a", "an", "and", "to", "of",
    "cut", "reduced", "lowered", "decreased",
    "tracked", "monitored", "kept", "maintained",
    "built", "produced", "created",
    "sent", "reported", "submitted", "entered", "logged",
    "quarter", "quarterly", "month", "monthly",
}

_NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?%?')
_DATE = re.compile(r'\b\d{1,2}/\d{4}\b')


def _norm(text: str) -> list[str]:
    """Lower-cased word tokens, punctuation dropped."""
    return re.findall(r"[a-z0-9%/]+", (text or "").lower())


def _content(tokens: list[str]) -> list[str]:
    """Tokens that carry evidence: everything not a permitted connective."""
    return [t for t in tokens if t not in _CONNECTIVES]


def _numbers(text: str) -> list[str]:
    return sorted(_NUMBER.findall(text or ""))


def _dates(text: str) -> list[str]:
    return sorted(_DATE.findall(text or ""))


@dataclass
class _Record:
    owner: str
    header_line: int
    header_text: str
    location: str = ""
    role: str = ""
    bullets: list[str] = None          # type: ignore[assignment]
    bullet_lines: list[int] = None     # type: ignore[assignment]


def _parse(draft: str) -> tuple[list[str], list[_Record]]:
    """Split a draft into lines and employer records with their line indices.

    Wrapped bullets are joined. Both the sources and the drafts are hard-wrapped at
    ~80 columns, so

        - Compared quarterly parts usage against the prior four quarters to set reorder
          points.

    is ONE bullet on TWO lines. Treating the continuation as a separate line silently
    truncated it to "…to set reorder", and a restoration built from that would have
    written an incomplete achievement into the résumé. Continuations are the lines
    after a bullet that are neither blank, nor a heading, nor a new record, nor a new
    bullet.
    """
    lines = draft.splitlines()
    records: list[_Record] = []
    current: _Record | None = None

    def is_continuation(index: int) -> bool:
        if index >= len(lines):
            return False
        candidate = lines[index]
        stripped = candidate.strip()
        if not stripped:
            return False
        if candidate.startswith('## ') or _HEAD.match(stripped) or _ROLE.match(stripped):
            return False
        return stripped.startswith('- ') is False and candidate[:1] in (' ', '\t')

    for index, line in enumerate(lines):
        if line.startswith('## '):
            current = None
            continue
        match = _HEAD.match(line.strip())
        if match:
            current = _Record(owner=match.group(1), header_line=index,
                              header_text=line.strip(), bullets=[], bullet_lines=[])
            if match.group(2):
                current.location = match.group(2).strip()
            records.append(current)
            continue
        if current is None:
            continue
        stripped = line.strip()
        if _ROLE.match(stripped) and not current.role:
            current.role = stripped
        elif stripped.startswith('- '):
            current.bullets.append(stripped[2:].strip())
            current.bullet_lines.append(index)
        elif current.bullet_lines and is_continuation(index):
            # Join onto the previous bullet with a single space.
            current.bullets[-1] = f"{current.bullets[-1]} {stripped}".strip()
    return lines, records


def _source_records(sources: dict[str, str]) -> dict[str, list[_Record]]:
    """Owner -> records, per source file. Empty list means the name was ambiguous."""
    out: dict[str, list[_Record]] = {}
    for text in sources.values():
        _lines, records = _parse(text)
        seen: dict[str, int] = {}
        for record in records:
            seen[record.owner] = seen.get(record.owner, 0) + 1
        for record in records:
            if seen[record.owner] > 1:
                out[record.owner] = []        # ambiguous, never usable
            else:
                out.setdefault(record.owner, [record])
    return out


def _paraphrase_only(correction: str, source_bullets: list[str]) -> bool:
    """True when `correction` says what some source bullet says, modulo connectives.

    Numbers and dates must match EXACTLY — a correction may not introduce or drop a
    figure or a date. Content words must be a multiset-subset of the source bullet's,
    so nothing may be added; removal is allowed because the check runs against the
    BEFORE text too (see `_removes_only_unsourced`).
    """
    corr_content = _content(_norm(correction))
    corr_numbers = _numbers(correction)
    for bullet in source_bullets:
        if _numbers(bullet) != corr_numbers:
            continue
        src_content = _content(_norm(bullet))
        # Every content token in the correction must be present in the source.
        remaining = list(src_content)
        ok = True
        for token in corr_content:
            if token in remaining:
                remaining.remove(token)
            else:
                ok = False
                break
        if ok:
            return True
    return False


def _bag(text: str) -> list[str]:
    return [t for t in _norm(text) if t not in _CONNECTIVES]


def _removes_only_unsourced(before: str, after: str, source_text: str,
                            elsewhere_in_draft: str = "") -> bool:
    """Removing content is allowed only when nothing source-backed is lost.

    Three cases, in order:

    1. The edit ADDS a token not present before → refuse. A correction may drop or
       reword, never introduce.
    2. A dropped token is not in any source → that is the invented clause → allow.
    3. A dropped token IS source-backed → allow ONLY if the same token still appears
       elsewhere in the draft, i.e. under its correct owner. This is what makes the
       wrong-record repair legal: "reconciled 1,180 fuel invoices" leaves the Cascade
       bullet but is already present verbatim in the Harborline bullet, so no fact is
       lost. Without this, every de-duplication would be refused as a loss.

    Connectives are excluded from the comparison on both sides: swapping "by" for
    "through" shows as one drop and one add, and neither is a claim about the
    candidate.
    """
    src = set(_content(_norm(source_text)))
    surviving = _bag(elsewhere_in_draft)

    remaining = _bag(after)
    dropped: list[str] = []
    for token in _bag(before):
        if token in remaining:
            remaining.remove(token)
        else:
            dropped.append(token)

    # "Added" means NOT IN THE SOURCE, not "not in the before text". A clause the
    # passage omitted but the source supports — "across the terminal fleet" here —
    # is restored evidence, not an invention. Requiring it to be absent would
    # reject every correction that reinstates sourced wording the defective draft
    # had dropped.
    invented = [token for token in remaining if token not in src]
    if invented:
        return False
    for token in dropped:
        if token not in src:
            continue                      # invented clause removed: correct
        if token in surviving:
            continue                      # fact survives under its correct owner
        return False                      # source-backed content destroyed
    return True


_CREDENTIAL = re.compile(
    r"\b(certified|certificate|certification|licen[cs]e|credential|"
    r"cics|chmm|cscp|osha)\b", re.IGNORECASE)


def _introduces_credential(correction: str, source_bullets: list[str]) -> bool:
    """True when the correction names a credential the owner's bullets do not.

    A qualification belongs to the Certifications section. Moving one into a work
    record changes what the record asserts about the role, so it is refused even when
    the credential itself is genuine.
    """
    if not _CREDENTIAL.search(correction or ""):
        return False
    haystack = " ".join(source_bullets).lower()
    for match in _CREDENTIAL.finditer(correction):
        if match.group(0).lower() not in haystack:
            return True
    return False


def _locate_bullet(draft: str, passage: str) -> tuple[int | None, str]:
    """Find the ONE line this passage refers to, or say why it cannot be found."""
    lines, _records = _parse(draft)
    target = " ".join(passage.split())
    hits = [i for i, line in enumerate(lines)
            if line.strip().startswith('- ')
            and " ".join(line.strip()[2:].split()) == target]
    if len(hits) == 1:
        return hits[0], "ok"
    if len(hits) > 1:
        return None, "ambiguous_target"
    # Presentation-tolerant fallback; still must be unique.
    located = dc.locate_passage(passage, draft)
    if located is None:
        return None, "passage_not_located"
    cand = [i for i, line in enumerate(lines)
            if " ".join(line.strip().lstrip('- ').split()) == " ".join(located.split())]
    if len(cand) == 1:
        return cand[0], "ok"
    return None, "ambiguous_target" if cand else "passage_not_located"


def _owner_of_index(records: list[_Record], index: int) -> _Record | None:
    prior = [r for r in records if r.header_line < index]
    return prior[-1] if prior else None


# ── The three repair kinds ──────────────────────────────────────────────────

def _repair_in_place(draft: str, sources: dict[str, str],
                     issue: dc.Issue) -> tuple[str, str]:
    """Replace, or delete, ONE bullet inside its own employer record."""
    lines, records = _parse(draft)
    index, why = _locate_bullet(draft, issue.passage or "")
    if index is None:
        return draft, why

    record = _owner_of_index(records, index)
    if record is None:
        return draft, "owner_unresolved"
    if sum(1 for r in records if r.owner == record.owner) != 1:
        return draft, "owner_ambiguous"

    src_records = _source_records(sources)
    owned = src_records.get(record.owner)
    if not owned:
        return draft, "owner_not_in_source"
    src_bullets = owned[0].bullets
    src_text = "\n".join(sources.values())

    correction = (issue.correction or "").strip()
    before = lines[index].strip()[2:].strip()

    # Everything the edit does not touch, so a de-duplication can show the fact
    # survives elsewhere under its correct owner.
    surviving = "\n".join(line for i, line in enumerate(lines) if i != index)

    if not correction or dc.is_omit_marker(correction):
        # Deletion of a whole bullet. Only permitted when nothing source-backed is
        # lost: the content must be absent from the sources, or survive elsewhere.
        if not _removes_only_unsourced(before, "", src_text, surviving):
            return draft, "deletion_removes_sourced_content"
        del lines[index]
        return "\n".join(lines) + ("\n" if draft.endswith("\n") else ""), "applied"

    new_text = correction
    if new_text.startswith('- '):
        new_text = new_text[2:]
    new_text = " ".join(new_text.split())

    if not _paraphrase_only(new_text, src_bullets):
        return draft, "replacement_not_source_equivalent"
    if not _removes_only_unsourced(before, new_text, src_text, surviving):
        return draft, "removal_drops_sourced_content"
    # A replacement cannot introduce a qualification into a work record. Checked
    # against the source's own bullets for this owner, so any credential text the
    # employer record does not contain is refused wherever it appears in the
    # correction.
    if _introduces_credential(new_text, src_bullets):
        return draft, "credential_moved_into_record"

    indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
    lines[index] = f"{indent}- {new_text}"
    return "\n".join(lines) + ("\n" if draft.endswith("\n") else ""), "applied"


def _repair_restore_record(draft: str, sources: dict[str, str],
                           issue: dc.Issue) -> tuple[str, str]:
    """Append a whole employer record, taking EVERY line from the source.

    The reviewer's correction supplies the DESTINATION and identifies the record; the
    CONTENT is read from the source file. That is why an invented date cannot survive:
    the corrected block is never taken from the model's prose.
    """
    src_records = _source_records(sources)
    correction = " ".join((issue.correction or "").split())
    if not correction:
        return draft, "restoration_without_target"

    owner = None
    for name, recs in src_records.items():
        if recs and name.lower() in correction.lower():
            owner = name
            break
    if owner is None:
        return draft, "restoration_owner_not_source_backed"
    if len(src_records[owner]) != 1:
        return draft, "restoration_owner_ambiguous"

    record = src_records[owner][0]
    if any(r.owner == owner for r in _parse(draft)[1]):
        return draft, "restoration_duplicate_owner"
    if not record.role:
        return draft, "restoration_missing_role"
    # A restoration may not smuggle a qualification into a work record. The block
    # is built from the source, but the reviewer's correction is what selected the
    # record, so it is checked here too.
    if _introduces_credential(correction, record.bullets):
        return draft, "credential_moved_into_record"

    # Reject a correction whose dates disagree with the source, rather than ignoring
    # them: a reviewer that proposes other dates has misread the record.
    if _dates(correction):
        if _dates(correction) != _dates(record.header_text + " " + record.role):
            return draft, "restoration_dates_disagree_with_source"

    block = ["", record.header_text]
    if record.role:
        block.append(record.role)
    block += [f"- {b}" for b in record.bullets]

    lines = draft.splitlines()
    # Destination: at the end of the Professional Experience section, immediately
    # before the next section heading if one follows. Blanks are normalised so the
    # insert cannot produce a doubled separator.
    cert = next((i for i, line in enumerate(lines)
                 if line.startswith('## Certifications')), None)
    if cert is not None:
        insert_at = cert
        while insert_at > 0 and not lines[insert_at - 1].strip():
            insert_at -= 1
        new_lines = lines[:insert_at] + block + ["", ""] + lines[cert:]
    else:
        new_lines = lines + block
    out = "\n".join(new_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).rstrip("\n") + "\n"
    return out, "applied"


# ── Entry point ─────────────────────────────────────────────────────────────

def apply_repair(draft: str, sources: dict[str, str],
                 issues: list[dc.Issue]) -> tuple[str, str]:
    """Apply ONE reviewer issue. Returns (document, status).

    One issue at a time is deliberate: the callers re-review the exact result, so a
    multi-issue application would make it impossible to attribute a later refusal to a
    specific edit. A caller with several issues applies them in sequence and
    re-reviews after each.
    """
    if not issues:
        return draft, "not_applicable"
    if len(issues) != 1:
        return draft, "ambiguous_or_multiple_issues"
    issue = issues[0]
    if issue.problem == dc.PROBLEM_OMITTED or \
            (issue.passage or "").strip() == dc.OMITTED_MARKER:
        return _repair_restore_record(draft, sources, issue)
    return _repair_in_place(draft, sources, issue)
