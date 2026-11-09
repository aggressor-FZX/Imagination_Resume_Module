#!/usr/bin/env python3
"""Produce a BLIND audit packet: outputs only, no configuration labels.

Why this exists
---------------
The instruction is that the model cannot be its own sole evaluator, and that an
independent human should judge the outputs **without knowing which configuration
produced them**. That is easy to say and easy to get wrong: the natural packet is
a table with a "mode" column, or files named `conservative-3.md`, either of which
tells the auditor which policy wrote what and invites them to grade the policy
rather than the document.

So this script:

* renames every output to an opaque label drawn from a seeded shuffle, so the
  mapping cannot be guessed from order or filename;
* strips any line in the output that names the configuration (mode, policy,
  skill-inference version, "imaginative"/"conservative");
* writes the label → configuration mapping to a SEPARATE file the auditor is told
  not to open, so the mapping survives for analysis without being available
  during judging;
* emits a judgement sheet with the verdict vocabulary the existing audit already
  uses (FAITHFUL / FAITHFUL (bridge) / ADDITION), so results stay comparable.

What this does NOT do
--------------------
It does not judge. It cannot: deciding whether a sentence states a claim the source
does not is exactly the semantic judgement a human must make. A model may assist
with sorting or highlighting, but the acceptance labels must not come from the
model that produced the output — that is the failure mode this file exists to
avoid.

Usage:
    python scripts/blind_audit.py --inputs results/*.md --sources SOURCE.md \
        --out audit/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import sys

# Any of these appearing in an output would identify the configuration.
_CONFIG_MARKERS = re.compile(
    r"(conservative|imaginative|skill_inference|policy_version|"
    r"bridge|mode\s*[:=])",
    re.IGNORECASE,
)

VERDICTS = ("FAITHFUL", "FAITHFUL (bridge)", "ADDITION", "OMITTED")


def scrub(text: str) -> tuple[str, int]:
    """Remove configuration-identifying lines. Returns (text, removed_count)."""
    kept, removed = [], 0
    for line in text.splitlines():
        if _CONFIG_MARKERS.search(line):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="generated documents, any configuration order")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="the candidate source documents the auditor may consult")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261109,
                    help="shuffle seed, so the packet is reproducible")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "outputs").mkdir(parents=True, exist_ok=True)

    paths = [pathlib.Path(p) for p in args.inputs]
    if not all(p.exists() for p in paths):
        missing = [str(p) for p in paths if not p.exists()]
        sys.exit(f"missing inputs: {missing}")

    # A seeded shuffle decides the labels, so neither the auditor nor a reader of
    # this script can infer the mapping from file order.
    order = list(range(len(paths)))
    random.Random(args.seed).shuffle(order)
    labels = [f"OUT-{i:02d}" for i in range(1, len(paths) + 1)]

    mapping = {}
    scrubbed_counts = {}
    for label, index in zip(labels, order):
        source_path = paths[index]
        text = source_path.read_text()
        clean, removed = scrub(text)
        scrubbed_counts[label] = removed
        (out / "outputs" / f"{label}.md").write_text(clean)
        mapping[label] = {
            "original": source_path.name,
            # Hash so the packet can be tied back later without trusting filenames.
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "config_markers_removed": removed,
        }

    # The key lives beside the packet, under a name that makes clear it must not
    # be opened during judging.
    (out / "DO_NOT_OPEN_label_key.json").write_text(json.dumps(mapping, indent=2))

    (out / "sources").mkdir(exist_ok=True)
    for path in map(pathlib.Path, args.sources):
        (out / "sources" / path.name).write_text(path.read_text())

    sheet = [
        "# Blind semantic audit",
        "",
        "Judge each `outputs/OUT-*.md` against `sources/` **without opening",
        "`DO_NOT_OPEN_label_key.json`**. The packet is deliberately unlabelled so",
        "the judgement is about the document, not the configuration.",
        "",
        "For every sentence that differs from the source, record one verdict:",
        "",
        *[f"- **{v}**" for v in VERDICTS],
        "",
        "A novel word is NOT an invented fact. The question is whether the "
        "sentence states something the source does not.",
        "",
        "| Output | Sentence | Source | Verdict | Why |",
        "|---|---|---|---|---|",
        "| | | | | |",
        "",
        "## Per-output summary",
        "",
        "| Output | Faithful | Faithful (bridge) | Addition | Omitted | Deliverable? |",
        "|---|---|---|---|---|---|",
        *[f"| {label} | | | | | |" for label in labels],
    ]
    (out / "JUDGEMENT_SHEET.md").write_text("\n".join(sheet) + "\n")

    print(f"packet written to {out}")
    print(f"  {len(labels)} outputs, shuffled with seed {args.seed}")
    removed = sum(scrubbed_counts.values())
    print(f"  {removed} configuration-identifying line(s) removed before labelling")
    if removed == 0:
        print("  NOTE: no config markers were found. If the inputs genuinely never")
        print("        name their mode, that is fine; if they were scrubbed already,")
        print("        confirm nothing else in the text reveals it.")
    print("  the label key is DO_NOT_OPEN_label_key.json — withhold it from the "
          "auditor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
