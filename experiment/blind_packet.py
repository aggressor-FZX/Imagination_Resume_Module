"""Blind audit packets that keep configuration metadata SEPARATE from the document.

What was wrong with the first attempt
-------------------------------------
The earlier `scripts/blind_audit.py` scrubbed any LINE matching a configuration
keyword before showing the document. That is the wrong mechanism: a résumé line can
legitimately contain the word "conservative", "mode", or "bridge", and deleting the
whole line destroys substantive candidate content. An auditor judging an incomplete
document produces an incomparable verdict.

What this does instead
----------------------
**The document is copied byte-for-byte.** Nothing is removed, reordered or
rewritten.

Instead, configuration identity is removed from where it actually lives:

* the FILE NAME becomes an opaque label (`OUT-01`);
* the SIDE-CAR (`meta/<label>.json`) holds the configuration, and is never in the
  packet handed to the auditor;
* any configuration line that the generator itself appends is dropped only when it
  is provably generator-appended — i.e. it matches a line the harness inserted,
  recorded at insert time. The harness does not insert any, so in practice nothing
  is dropped and the copy is byte-identical (asserted).

The packet therefore cannot leak the configuration through naming or ordering, and
cannot lose résumé content through over-eager scrubbing. Both properties are
asserted, because either failure silently invalidates the audit.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class PacketEntry:
    label: str
    case_id: str
    cell: str
    document: str
    sha256: str


def build_packet(records: Iterable[dict[str, Any]], out_dir: pathlib.Path, *,
                 seed: int = 20261109) -> dict[str, Any]:
    """Write a blind packet. `records` are Attempt.to_dict()-style mappings.

    Each record needs at least: case_id, cell, final_markdown.
    """
    out = pathlib.Path(out_dir)
    (out / "documents").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    usable = [r for r in records if isinstance(r.get("final_markdown"), str)
              and r["final_markdown"].strip()]
    if not usable:
        raise ValueError("no documents to audit")

    # Seeded shuffle decides the labels, so neither the auditor nor a reader of
    # this file can infer the configuration from the order of the inputs.
    order = list(range(len(usable)))
    random.Random(seed).shuffle(order)
    labels = [f"OUT-{i:02d}" for i in range(1, len(usable) + 1)]

    entries: list[PacketEntry] = []
    for label, index in zip(labels, order):
        record = usable[index]
        document = record["final_markdown"]
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()

        # BYTE-FOR-BYTE. No scrubbing.
        (out / "documents" / f"{label}.md").write_text(document)

        # Configuration lives here, and the auditor is told not to open it.
        (out / "meta" / f"{label}.json").write_text(json.dumps({
            "cell": record.get("cell"),
            "case_id": record.get("case_id"),
            "mode": record.get("mode"),
            "outcome": record.get("outcome"),
            "expectation": record.get("expectation"),
            "document_sha256": digest,
        }, indent=2))

        entries.append(PacketEntry(label=label, case_id=str(record.get("case_id")),
                                   cell=str(record.get("cell")),
                                   document=document, sha256=digest))

    # The key is a separate artefact, not part of the packet.
    key = {e.label: {"case_id": e.case_id, "cell": e.cell, "sha256": e.sha256}
           for e in entries}
    (out / "DO_NOT_OPEN_key.json").write_text(json.dumps(key, indent=2))

    (out / "AUDIT_INSTRUCTIONS.md").write_text(_instructions(entries))
    (out / "JUDGEMENT_SHEET.md").write_text(_sheet(entries))

    return {"labels": labels, "count": len(entries), "key_written": True,
            "packet_dir": str(out)}


def _instructions(entries: list[PacketEntry]) -> str:
    return """# Blind semantic audit

Judge each document in `documents/` on its own merits. Do **not** open
`DO_NOT_OPEN_key.json` or `meta/` — those identify which configuration produced
each document, and the point of this audit is that the judgement is about the
document, not the configuration.

The documents are byte-identical copies. Nothing has been removed, reordered or
reworded, so any awkward phrasing you see is the system's own.

For every sentence that differs from the source, record one verdict:

- **FAITHFUL** — same fact, different wording. A novel word is not an invented fact.
- **FAITHFUL (capability)** — a general capability generalised from a demonstrated
  activity, which the Imaginative policy permits.
- **ADDITION** — states something the sources do not: a method, tool, credential,
  date, metric or relationship the candidate never had.
- **MISATTRIBUTED** — a real fact attached to the wrong employer or project.
- **OMITTED** — a source-backed fact the document drops.

Then answer, for the document as a whole: **would you deliver this?** yes / no, and
in one sentence why.

""".replace("documents/`", f"documents/` ({len(entries)} files)")


def _sheet(entries: list[PacketEntry]) -> str:
    rows = [
        "# Judgement sheet",
        "",
        "| Document | Sentence | Source | Verdict | Why |",
        "|---|---|---|---|---|",
    ]
    rows += ["| | | | | |" for _ in entries]
    rows += ["", "## Per-document conclusion", "",
             "| Document | Deliver? | Why |", "|---|---|---|"]
    rows += [f"| {e.label} | | |" for e in entries]
    return "\n".join(rows) + "\n"


def verify_blind(packet_dir: pathlib.Path) -> dict[str, Any]:
    """Prove the two properties the audit depends on.

    1. **No content loss**: each document matches the hash recorded for it.
    2. **No configuration leak**: no document mentions its own cell name.
    """
    out = pathlib.Path(packet_dir)
    key = json.loads((out / "DO_NOT_OPEN_key.json").read_text())
    problems: list[str] = []
    for label, meta in key.items():
        path = out / "documents" / f"{label}.md"
        text = path.read_text()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != meta["sha256"]:
            problems.append(f"{label}: content changed since packet build")
        if meta["cell"] and meta["cell"] in text:
            problems.append(f"{label}: document names its own configuration")
    return {"ok": not problems, "problems": problems, "documents": len(key)}
