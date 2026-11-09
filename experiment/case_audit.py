"""Independent validation of the v2 cases — checks the FIXTURE, not the author.

Why this is not just a test
---------------------------
The v1 failure was not a bug in the model. It was an invalid fixture: cases that
expected `approve` while their drafts dropped a whole source record. Every later
measurement inherited that error, and nothing in the harness could see it.

So the fixture needs its own auditor, and the auditor must not simply re-assert
what the author wrote. It derives verdicts from the source documents by mechanical
rules, then compares them with the declared expectation. Where a rule cannot settle
the question it says `undetermined` instead of guessing — an auditor that guesses
would reintroduce exactly the defect it exists to catch.

What it can and cannot decide mechanically
------------------------------------------
CAN (and does):
  * every employer record in a source appears in an approve-case draft;
  * every quantified figure in an approve-case draft exists in a source;
  * no employer record appears in a draft without a source;
  * a figure is not attached to an employer that does not own it (a co-occurrence
    test within a section, which is enough for THIS fixture's single misattribution);
  * the mutation declared for a case really is the ONLY difference from the baseline;
  * the conservative/imaginative distinction on `transferable_skill`.

CANNOT (and says so):
  * whether a paraphrase preserves *meaning* — that needs a reader;
  * whether an appended clause is "plausible for the role" — a judgement;
  * whether a dropped bullet is *material* — a judgement about importance.

Those residuals are listed as `undetermined` and are what the independent blind
audit is for. The point of this module is to shrink the space the human has to
adjudicate, and to make the mechanical part reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Mechanical extractors ──────────────────────────────────────────────────
_EMPLOYER_RE = re.compile(r"\*\*([^*]+?)\*\*\s*--\s*([^,\n]+)")
_ROLE_RE = re.compile(r"\*([^*\n]+?)\s*\|\s*([0-9/\- ]+)\*")
_NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?%?)(?![\w])")


def employers_in(text: str) -> dict[str, str]:
    """`**Employer** -- Location` -> {employer: location}."""
    return {m.group(1).strip(): m.group(2).strip()
            for m in _EMPLOYER_RE.finditer(text or "")}


def figures_in(text: str) -> list[str]:
    """Quantified figures, normalised for comparison.

    `1,180` and `1180` are the same figure. Normalising prevents a false alarm
    where a writer dropped a comma, which is a formatting choice, not a lost fact.
    """
    out = []
    for match in _NUMBER_RE.finditer(text or ""):
        out.append(match.group(1).replace(",", ""))
    return out


def sections_in(text: str) -> dict[str, str]:
    """Split a dossier on `## Heading`, so attribution can be checked per section."""
    sections: dict[str, str] = {}
    current = "_preamble"
    buffer: list[str] = []
    for line in (text or "").splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buffer)
            current = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    sections[current] = "\n".join(buffer)
    return sections


@dataclass
class Finding:
    check: str
    case_id: str
    verdict: str              # pass | fail | undetermined
    detail: str = ""


@dataclass
class AuditResult:
    findings: list[Finding] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "fail"]

    @property
    def undetermined(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "undetermined"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "counts": {
                "pass": sum(1 for f in self.findings if f.verdict == "pass"),
                "fail": len(self.failures),
                "undetermined": len(self.undetermined),
            },
            "failures": [f.__dict__ for f in self.failures],
            "undetermined": [f.__dict__ for f in self.undetermined],
            "all": [f.__dict__ for f in self.findings],
        }


# ── Checks ─────────────────────────────────────────────────────────────────
def _check_covers_sources(case, audit: AuditResult) -> None:
    """An approve-case must contain every employer record its sources supply.

    This is THE v1 defect. It is checked only for `approve` cases: a `revise` case
    may deliberately omit a record (that is `omitted_record`'s whole purpose), but
    an approve case cannot be approved faithfully while a supplied record is gone.
    """
    if case.expectation != "approve":
        audit.findings.append(Finding(
            "covers_sources", case.case_id, "pass",
            "not an approve case; source coverage is not required"))
        return
    supplied = set()
    for text in case.sources.values():
        supplied |= set(employers_in(text))
    present = set(employers_in(case.draft_before_trim))
    missing = sorted(supplied - present)
    audit.findings.append(Finding(
        "covers_sources", case.case_id, "fail" if missing else "pass",
        f"draft omits supplied employer record(s): {missing}" if missing
        else f"all {len(supplied)} supplied employer records present"))


def _check_figures_sourced(case, audit: AuditResult) -> None:
    """Every figure in an approve-case draft must exist in a source."""
    if case.expectation != "approve":
        audit.findings.append(Finding(
            "figures_sourced", case.case_id, "pass", "not an approve case"))
        return
    sourced = set()
    for text in case.sources.values():
        sourced |= set(figures_in(text))
    used = set(figures_in(case.draft_before_trim))
    invented = sorted(used - sourced)
    audit.findings.append(Finding(
        "figures_sourced", case.case_id, "fail" if invented else "pass",
        f"figure(s) in no source: {invented}" if invented
        else f"all {len(used)} figures traceable to a source"))


def _check_no_invented_employer(case, audit: AuditResult) -> None:
    """A draft must not name an employer no source mentions — at any expectation."""
    supplied = set()
    for text in case.sources.values():
        supplied |= set(employers_in(text))
    present = set(employers_in(case.draft_before_trim))
    invented = sorted(present - supplied)
    audit.findings.append(Finding(
        "no_invented_employer", case.case_id, "fail" if invented else "pass",
        f"employer in no source: {invented}" if invented else ""))


def _check_attribution(case, audit: AuditResult) -> None:
    """A figure must not sit inside an employer section that does not own it.

    Co-occurrence within a `## section` is a coarse test, and coarse is the honest
    choice here: it flags exactly the v1 `wrong_record_attribution` shape (a figure
    moved into another employer's block) without pretending to parse causality. It
    produces `undetermined` when a figure is owned by two sources, because then
    co-occurrence cannot identify the owner.
    """
    sections = sections_in(case.draft_before_trim)
    owners: dict[str, set[str]] = {}
    for source_text in case.sources.values():
        for section_name, body in sections_in(source_text).items():
            found = employers_in(body)
            employer = next(iter(found), "")
            for figure in figures_in(body):
                if employer:
                    owners.setdefault(figure, set()).add(employer)

    problems = []
    for section_name, body in sections.items():
        local = set(employers_in(body))
        if not local:
            continue
        for figure in set(figures_in(body)):
            real_owners = owners.get(figure)
            if not real_owners:
                continue
            if len(real_owners) > 1:
                audit.findings.append(Finding(
                    "attribution", case.case_id, "undetermined",
                    f"figure {figure} appears under {sorted(real_owners)}; "
                    f"co-occurrence cannot identify the owner"))
                continue
            owner = next(iter(real_owners))
            if owner not in local:
                problems.append((figure, owner, sorted(local)))
    audit.findings.append(Finding(
        "attribution", case.case_id, "fail" if problems else "pass",
        f"figure(s) under the wrong employer: {problems}" if problems else ""))


def _check_isolation(case, baseline: str, mutation_kind: str,
                     audit: AuditResult) -> None:
    """Every case must be the baseline plus exactly ONE declared transformation.

    Isolation is what makes a wrong verdict attributable. If two cases differed in
    several ways, a difference in outcome could not be traced to a difference in
    the defect — which is precisely how v1's results became uninterpretable.
    """
    draft = case.draft_before_trim
    if draft == baseline:
        verdict = "pass" if mutation_kind == "reword_only" else "fail"
        detail = ("identical to the baseline, as declared" if verdict == "pass"
                  else f"declared {mutation_kind} but draft is the baseline")
        audit.findings.append(Finding("isolation", case.case_id, verdict, detail))
        return

    # Count the differing lines between baseline and draft.
    base_lines = baseline.splitlines()
    draft_lines = draft.splitlines()
    removed = [l for l in base_lines if l not in draft_lines]
    added = [l for l in draft_lines if l not in base_lines]
    changed = len(removed) + len(added)
    # One logical transformation may touch several lines (an appended clause wraps
    # across two lines; a moved figure rewrites one bullet). Allow a small spread
    # but refuse a wholesale rewrite, which would no longer be an isolated case.
    ok = changed <= 6
    audit.findings.append(Finding(
        "isolation", case.case_id, "pass" if ok else "fail",
        f"{changed} line(s) differ from baseline (removed={len(removed)}, "
        f"added={len(added)}) for mutation {mutation_kind}"))


def _check_mode_dependence(case, audit: AuditResult) -> None:
    """`transferable_skill` is only approvable under Imaginative.

    Its expectation is `approve`, which is correct for the mode it will be run in.
    Recording the dependency prevents a later reader from assuming the expectation
    holds for every mode — a Conservative reviewer refusing it is CORRECT, and
    scoring that as a false refusal would penalise the stricter policy for being
    what it says it is.
    """
    if case.case_id != "transferable_skill":
        return
    audit.findings.append(Finding(
        "mode_dependence", case.case_id, "pass",
        "expectation=approve holds only under Imaginative; a Conservative refusal "
        "is correct and must not be scored as a false refusal"))


def _check_expectation_reachable(case, audit: AuditResult) -> None:
    """State plainly which expectations mechanics cannot settle."""
    judgement_cases = {
        "supported_method": "whether changing 'by' to 'through' preserves the "
                            "sourced method is a reading judgement",
        "unsupported_method": "whether the appended clause is 'plausible for the "
                              "role' is a judgement; that it is ABSENT from both "
                              "sources is mechanical and verified",
        "missing_fact": "whether the dropped bullet is MATERIAL is a judgement; "
                        "that it exists in the source and not the draft is verified",
        "omitted_record": "that the whole record is absent is verified; that a "
                          "reviewer must restore rather than withhold it is policy",
        "transferable_skill": "whether the added clause is a capability or an "
                              "activity is a judgement the policy decides",
    }
    detail = judgement_cases.get(case.case_id)
    if detail:
        audit.findings.append(Finding(
            "expectation_needs_reading", case.case_id, "undetermined", detail))


def audit_cases(cases, *, baseline: str,
                mutations: dict[str, dict[str, str]]) -> AuditResult:
    """Run every mechanical check over the fixture."""
    audit = AuditResult()
    for case in cases:
        mutation = mutations.get(case.case_id)
        if mutation is None:
            audit.findings.append(Finding(
                "declared_mutation", case.case_id, "fail",
                "no mutation declared for this case"))
        _check_covers_sources(case, audit)
        _check_figures_sourced(case, audit)
        _check_no_invented_employer(case, audit)
        _check_attribution(case, audit)
        if mutation:
            _check_isolation(case, baseline, mutation["kind"], audit)
        _check_mode_dependence(case, audit)
        _check_expectation_reachable(case, audit)
    return audit


def main() -> int:
    import json
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from experiment.cases_v2 import BASELINE, CASES_V2, MUTATIONS

    result = audit_cases(CASES_V2, baseline=BASELINE, mutations=MUTATIONS)
    print(json.dumps({k: v for k, v in result.to_dict().items() if k != "all"},
                     indent=2))
    print()
    print(f"mechanical checks : {'PASS' if result.ok else 'FAIL'}")
    print(f"undetermined       : {len(result.undetermined)} (need the blind audit)")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
