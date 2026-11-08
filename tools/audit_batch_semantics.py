#!/usr/bin/env python3
"""Semantic audit of the fixed-build batch against the frozen sources.

This is an AUDIT, not a gate. It does not change production behaviour and it is
not a lexical verifier: a novel word is not an invented fact. "repeat" against a
source that says "failed repeatedly" is the same fact; "year-over-year" may
faithfully describe "compared each month against the same month in the prior
year". Judging those requires reading the sentence, which is what this tool
records for a human to decide.

What it does mechanically, because these ARE decidable:

  * OMISSIONS — every required fact in the ledger must appear in the output,
    under its own owner. A missing fact, or a fact attached to the wrong
    employer, is a defect regardless of wording.
  * OWNERSHIP — each employer keeps its own role, dates and metrics. A metric
    that moves to another employer is a defect.
  * PROHIBITED — a fact the ledger marks prohibited (an employer, credential or
    metric absent from the source) must not appear.
  * ADDITIONS — a sentence whose content is not accounted for by the ledger is
    recorded in full, with the source evidence, for semantic judgment. It is
    NOT classified as a defect here.

Output is a JSON artifact plus a readable report. Every disputed sentence is
recorded verbatim with the source lines it could be restating, so the judgment
can be made from the artifact alone.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = HERE / "tests" / "fixtures"
CAPTURES = HERE / "captures" / "recovery-validation"


# ---------------------------------------------------------------------------
# Ledgers. Derived by reading the frozen source documents only.
# ---------------------------------------------------------------------------

BRIDGE_LEDGER = {
    "fixture": "bridge_opportunity",
    "employers": [
        {"name": "Harborline Transit Authority", "role": "Fuel Reporting Analyst",
         "dates": "04/2019 - Present", "location": "Tacoma, WA",
         "metrics": ["14 depots", "1,180 fuel invoices"],
         "achievements": [
             "Tracked monthly fuel consumption across 14 depots and compared each "
             "month against the same month in the prior year",
             "Built the depot scorecard that supervisors review in the Monday "
             "operations meeting",
             "Reconciled 1,180 fuel invoices against depot dispensing records "
             "each quarter",
         ]},
        {"name": "Cascade Freight Partners", "role": "Cost Reporting Coordinator",
         "dates": "07/2015 - 03/2019", "location": "Kent, WA",
         "metrics": ["3 years"],
         "achievements": [
             "Audited 3 years of maintenance cost records to find which equipment "
             "failed repeatedly",
             "Prepared the monthly cost summary reviewed by the regional manager",
             "Rebuilt the shipment delay log that the dispatch team relies on",
         ]},
        {"name": "Ridgeline Equipment Services", "role": "Inventory Analyst",
         "dates": "08/2023 - Present", "location": "Fife, WA",
         "metrics": ["7 service branches", "2 branches", "prior four quarters"],
         "achievements": [
             "Maintain parts availability records for 7 service branches",
             "Flagged slow-moving stock and proposed the reduction plan adopted "
             "for 2 branches",
             "Compared quarterly parts usage against the prior four quarters to "
             "set reorder points",
         ]},
    ],
    "education": [
        {"institution": "Puget Sound Technical College", "award":
         "Associate of Applied Science, Business Analytics", "year": "2015"},
        {"institution": "Inland Records Institute", "award":
         "Certificate, Inventory Control", "year": "2022"},
    ],
    "certifications": ["Certified Production and Inventory Management (CPIM)"],
    "summary_claims": [
        "Operations analyst with 9 years of experience in depot reporting and "
        "fuel cost control across public transit operations",
        "Operations analyst supporting stock planning and parts availability for "
        "a regional equipment service network",
    ],
    "prohibited": [],
}

REALISTIC_LEDGER = {
    "fixture": "realistic_candidate",
    "employers": [
        {"name": "Northline Fabrication, Inc.", "role":
         "Welder / Maintenance Helper", "dates": "03/2012 - 09/2015",
         "location": "Spokane, WA", "metrics": ["14 industrial contracts"],
         "achievements": [
             "Fabricated and repaired structural steel assemblies for 14 "
             "industrial contracts",
             "Performed preventive maintenance on shop welding and cutting "
             "equipment",
         ]},
        {"name": "Kettle Falls Electrical Contractors", "role":
         "Electrical Apprentice", "dates": "10/2015 - 06/2019",
         "location": "Spokane, WA",
         "metrics": ["31 commercial buildouts", "8,000 apprenticeship hours"],
         "achievements": [
             "Pulled and terminated branch circuits on 31 commercial buildouts",
             "Completed 8,000 apprenticeship hours under a licensed journeyman",
         ]},
        {"name": "Pend Oreille Power Systems", "role": "Journeyman Electrician",
         "dates": "07/2019 - 02/2023", "location": "Spokane Valley, WA",
         "metrics": ["9 substations", "4.5 hours", "2.75 hours", "4 apprentices"],
         "achievements": [
             "Diagnosed and repaired switchgear faults at 9 substations",
             "Cut average fault-response time from 4.5 hours to 2.75 hours",
             "Trained 4 apprentices on lockout/tagout procedures",
         ]},
        {"name": "Riverbend Facilities Group", "role": "Maintenance Electrician",
         "dates": "03/2023 - Present", "location": "Coeur d'Alene, ID",
         "metrics": ["6 commercial properties", "27%"],
         "achievements": [
             "Maintain electrical distribution and BAS controls for 6 commercial "
             "properties",
             "Reduced after-hours callouts by 27% by adding thermographic "
             "inspections",
         ]},
    ],
    "education": [
        {"institution": "Spokane Falls Institute of Technology", "award":
         "Associate of Applied Science, Electrical Systems Technology",
         "year": "2015"},
        {"institution": "Inland Northwest Safety Academy", "award":
         "Certificate, NFPA 70E Arc Flash Standards", "year": "2021"},
    ],
    "certifications": ["Washington 01 Journey Level Electrician License"],
    "summary_claims": [
        "Journeyman electrician with 12 years of electrical experience spanning "
        "commercial construction, industrial maintenance, and facility operations",
        "Maintenance electrician focused on electrical distribution and building "
        "automation systems in occupied commercial properties",
    ],
    # P2 of the review package: in the posting, absent from the record.
    "prohibited": ["motor controls"],
}

LEDGERS = {"bridge_opportunity": BRIDGE_LEDGER,
           "realistic_candidate": REALISTIC_LEDGER}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    return " ".join((text or "").replace("SYNTHETIC-ONLY", "").split()).casefold()


def source_text(fixture: str) -> str:
    F = FIXTURES / fixture
    return ((F / "candidate-main.txt").read_text() + "\n\n---\n\n"
            + (F / "candidate-supplement.txt").read_text())


def output_lines(markdown: str) -> list[str]:
    """Substantive output lines, with the Markdown bullet marker removed."""
    out = []
    for raw in (markdown or "").splitlines():
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        body = re.sub(r"^[-*+]\s+", "", t).strip()
        if body:
            out.append(body)
    return out


def employer_of(markdown: str, needle: str, employers: list[dict]) -> str | None:
    """The employer whose record contains ``needle``, by nearest preceding head."""
    position = markdown.find(needle)
    if position < 0:
        return None
    best, best_at = None, -1
    for emp in employers:
        at = markdown.rfind(emp["name"], 0, position)
        if at > best_at:
            best, best_at = emp["name"], at
    return best


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_omissions(markdown: str, ledger: dict) -> list[dict]:
    """Every required fact must appear, under its own owner."""
    flat = norm(markdown)
    findings = []
    for emp in ledger["employers"]:
        if norm(emp["name"]) not in flat:
            findings.append({"kind": "omission", "fact": emp["name"],
                             "detail": "employer absent from the output"})
            continue
        if norm(emp["role"]) not in flat:
            findings.append({"kind": "omission", "fact": emp["role"],
                             "detail": f"role absent for {emp['name']}"})
        for metric in emp["metrics"]:
            if norm(metric) not in flat:
                findings.append({"kind": "omission", "fact": metric,
                                 "detail": f"metric absent for {emp['name']}"})
    for edu in ledger["education"]:
        if norm(edu["institution"]) not in flat:
            findings.append({"kind": "omission", "fact": edu["institution"],
                             "detail": "education record absent"})
    for cert in ledger["certifications"]:
        if norm(cert) not in flat:
            findings.append({"kind": "omission", "fact": cert,
                             "detail": "certification absent"})
    return findings


def check_ownership(markdown: str, ledger: dict) -> list[dict]:
    """A metric must sit under the employer that owns it."""
    findings = []
    for emp in ledger["employers"]:
        for metric in emp["metrics"]:
            owner = employer_of(markdown, metric, ledger["employers"])
            if owner is None:
                continue                      # absence is an omission, not this
            if owner != emp["name"]:
                findings.append({"kind": "ownership", "fact": metric,
                                 "expected": emp["name"], "found": owner,
                                 "detail": f"{metric} appears under {owner}"})
    return findings


def check_prohibited(markdown: str, ledger: dict) -> list[dict]:
    flat = norm(markdown)
    return [{"kind": "prohibited", "fact": term,
             "detail": "prohibited term present in the output"}
            for term in ledger["prohibited"] if norm(term) in flat]


def check_markers(markdown: str) -> list[dict]:
    return ([{"kind": "marker", "fact": "SYNTHETIC-ONLY",
              "detail": "provenance marker reached the output"}]
            if "SYNTHETIC-ONLY" in (markdown or "") else [])


def disputed_sentences(markdown: str, ledger: dict, source: str) -> list[dict]:
    """Sentences whose content the ledger does not account for.

    Recorded for SEMANTIC judgment, never classified as a defect. Each entry
    carries the sentence verbatim and the source lines that could be its basis,
    so the judgment can be made from the artifact.
    """
    src_lines = [l.strip() for l in source.splitlines() if l.strip()]
    ledger_text = norm(" ".join(
        [a for e in ledger["employers"] for a in e["achievements"]]
        + ledger["summary_claims"]
        + [e["award"] for e in ledger["education"]]
        + ledger["certifications"]))
    src_flat = norm(source)
    out = []
    for body in output_lines(markdown):
        # An employer/institution head is structure, and it is in the source
        # verbatim. It is not a claim to judge. The shape is
        # "**Name** -- Location", so it does not END with "**".
        if re.match(r"^\*\*[^*]+\*\*", body):
            continue
        if re.match(r"^\*[^*]+\*$", body):
            continue
        # A line that IS a source line is accounted for, whatever words it
        # contains: "stock level planning" is the source's own phrase.
        if norm(body) in src_flat:
            continue
        if len(body.split()) < 4:
            continue
        # A line whose every content word is already in the ledger is accounted
        # for; anything else is recorded for judgment.
        words = {w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", norm(body))
                 if len(w) > 3}
        novel = {w for w in words if w not in ledger_text}
        if not novel:
            continue
        # The source lines that share the most words with this sentence.
        scored = sorted(
            src_lines,
            key=lambda l: -len(words & {w for w in
                                        re.findall(r"[A-Za-z][A-Za-z'-]+", norm(l))
                                        if len(w) > 3}))
        out.append({"sentence": body, "novel_words": sorted(novel),
                    "source_evidence": [l.replace("SYNTHETIC-ONLY ", "")
                                        for l in scored[:2] if l.strip()]})
    return out


def audit(label: str, fixture: str) -> dict:
    path = CAPTURES / label / "response.json"
    if not path.exists():
        return {"label": label, "error": "no response.json"}
    body = json.loads(path.read_text())
    ledger = LEDGERS[fixture]
    markdown = body.get("rewritten_resume") or ""
    status = body.get("correction_status")
    record = {
        "label": label, "fixture": fixture,
        "mode": "conservative" if "conservative" in label else "imaginative",
        "status": status, "reason": body.get("correction_reason"),
        "bridges": [b.get("skill") for b in (body.get("skill_bridges") or [])],
    }
    if status != "approved" or not markdown:
        record["note"] = "withheld; no output to audit"
        return record
    source = source_text(fixture)
    record["omissions"] = check_omissions(markdown, ledger)
    record["ownership"] = check_ownership(markdown, ledger)
    record["prohibited"] = check_prohibited(markdown, ledger)
    record["markers"] = check_markers(markdown)
    record["disputed"] = disputed_sentences(markdown, ledger, source)
    record["defects"] = (len(record["omissions"]) + len(record["ownership"])
                         + len(record["prohibited"]) + len(record["markers"]))
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="finalB-")
    parser.add_argument("--out", default="captures/final/batch/audit.json")
    args = parser.parse_args(argv)

    records = []
    for d in sorted(CAPTURES.iterdir()):
        if not d.name.startswith(args.prefix):
            continue
        fixture = ("bridge_opportunity" if "bridge" in d.name
                   else "realistic_candidate")
        records.append(audit(d.name, fixture))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))

    approved = [r for r in records if r.get("status") == "approved"]
    clean = [r for r in approved if r.get("defects") == 0]
    print(f"runs audited: {len(records)}")
    print(f"approved: {len(approved)}")
    print(f"approved with zero mechanical defects: {len(clean)}")
    print()
    for r in records:
        if r.get("status") != "approved":
            print(f"{r['label']:48s} withheld  {r.get('reason')}")
            continue
        print(f"{r['label']:48s} defects={r['defects']:2d} "
              f"disputed={len(r['disputed']):2d}")
        for f in r["omissions"] + r["ownership"] + r["prohibited"] + r["markers"]:
            print(f"      {f['kind'].upper():11s} {f['fact']} — {f['detail']}")
    print()
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
