"""Controlled answers for provider phases a recording never performed.

Every capture from delivery runs 4-7 contains exactly two exchanges:
``source_inventory`` and ``review``. The live runs all stopped at the same point, so
none of them records a ``repair``, a ``post_patch_review``, a ``restore`` or a
``post_restore_review``. Replaying new code against those bytes therefore proves the
new code *starts* a repair; it cannot prove the repair is applied, re-reviewed,
validated and published, because the recording runs out.

This module supplies the missing answers. It is scaffolding, and it is labelled as
scaffolding everywhere it is used:

* the answers live in tests, are keyed by phase, and are never presented as model
  output;
* ``live_capture.ControlledExtension`` refuses to overwrite a recorded phase, so the
  recording stays authoritative for everything it actually contains;
* a controlled answer is written by the person who also wrote the answer key, so it
  cannot independently corroborate that key. What a controlled run proves is that the
  *mechanism* carries a defect to repair and then validates the result — never that a
  real model would answer this way.

The wire shape is OpenRouter's, because the adapter validates it strictly: the
envelope must carry the expected model, exactly one choice, ``finish_reason``
``stop``, and a JSON-string ``content``. ``_success`` builds that envelope.
"""
from __future__ import annotations

import json

MODEL = "qwen/qwen3-coder"


def success(content: dict, *, model: str = MODEL, provider: str = "controlled-test") -> str:
    """Wrap a protocol payload in the envelope the adapter validates."""
    return json.dumps({
        "id": "controlled-0",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "provider": provider,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(content)},
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def segmented_inventory(source_text: str, specials: tuple[str, ...] = (),
                        source_id: str = "document-0") -> list[dict]:
    """Partition a source into contiguous line-based items.

    The engine requires the required items to cover each source exactly, which is what
    makes "represented vs missing" well defined. Items follow source line boundaries so
    one item never straddles two unrelated facts. Each ``special`` substring is forced
    to be its own item, so a test can drive it being absent.

    ``source_id`` must match the document id in the capture, because every item id and
    citation is checked against the source map. Defaulting it here rather than in the
    caller keeps the two from drifting silently: a wrong id produces an inventory the
    engine rejects rather than one it quietly mis-attributes.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    for segment in source_text.splitlines(keepends=True):
        begin, end = offset, offset + len(segment)
        offset = end
        cuts = sorted({begin + segment.index(s) for s in specials if s in segment})
        cursor = begin
        for cut in cuts:
            if cut > cursor:
                spans.append((cursor, cut))
            match = next(s for s in specials if segment[cursor - begin:].find(s) >= 0
                         and begin + segment.index(s) == cut)
            spans.append((cut, cut + len(match)))
            cursor = cut + len(match)
        if cursor < end:
            spans.append((cursor, end))
    items = []
    for left, right in spans:
        if right <= left:
            continue
        items.append({
            "id": f"{source_id}:{left}:{right}",
            "source_id": source_id,
            "start": left,
            "end": right,
            "quote": source_text[left:right],
            "classification": "required",
        })
    return items


def inventory_answer(items: list[dict]) -> dict:
    return {"source_claims": [dict(item) for item in items]}


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

def _cite(item: dict) -> list[dict]:
    return [{"source_id": item["source_id"], "start": item["start"],
             "end": item["end"], "quote": item["quote"]}]


def _claim(line_id: str, start: int, end: int, verdict: str, citations=None) -> dict:
    return {"id": f"{line_id}:{start}:{end}", "line_id": line_id, "start": start,
            "end": end, "verdict": verdict, "citations": citations or []}


class ReviewScript:
    """Assemble a valid review answer from per-line decisions.

    The engine requires every line of the draft to be judged and every source item to
    be accounted for as represented or missing. This helper takes a mapping of line
    text -> decision so a test can state only the interesting lines:

        * ``("supported", needle)``   the line is supported by the source item
                                      containing ``needle``
        * ``("unsupported", None)``   the line claims something the source does not
                                      establish
        * ``("structural", None)``    heading or formatting

    Any item not claimed by a supported line is reported missing, which is what the
    engine reads as an omitted fact.
    """

    def __init__(self, items: list[dict]):
        self.items = items

    def _item_for(self, needle: str) -> dict | None:
        return next((i for i in self.items if needle in i["quote"]), None)

    def build(self, lines: list[dict], decisions: dict[str, tuple[str, str | None]]) -> dict:
        claims: list[dict] = []
        for line in lines:
            text, lid = line["text"], line["line_id"]
            key = next((k for k in decisions if k in text), None)
            if key is None:
                raise AssertionError(f"no decision for line {lid}: {text!r}")
            verdict, needle = decisions[key]
            if verdict == "supported":
                item = self._item_for(needle)
                if item is None:
                    raise AssertionError(f"no source item contains {needle!r}")
                claims.append(_claim(lid, 0, len(text), "supported", _cite(item)))
            else:
                claims.append(_claim(lid, 0, len(text), verdict, None))

        reps = []
        for item in self.items:
            span = None
            for claim in claims:
                if claim["verdict"] != "supported":
                    continue
                if any(c["start"] == item["start"] and c["end"] == item["end"]
                       for c in claim["citations"]):
                    span = {"line_id": claim["line_id"], "start": claim["start"],
                            "end": claim["end"]}
                    break
            reps.append({"source_claim_id": item["id"],
                         "status": "represented" if span else "missing",
                         "spans": [span] if span else []})
        return {"claims": claims, "representations": reps}

    def unrepresented(self, lines: list[dict], verdict: str = "unsupported") -> dict:
        """Every line judged, every item missing.

        Used for the negative controls where the point is that *nothing* is
        substantiated: the engine must still withhold, and must withhold for the right
        reason.
        """
        claims = [_claim(line["line_id"], 0, len(line["text"]), verdict, None)
                  for line in lines]
        reps = [{"source_claim_id": item["id"], "status": "missing", "spans": []}
                for item in self.items]
        return {"claims": claims, "representations": reps}


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def repair_answer(lines: list[dict], flagged_claim_ids: list[str],
                  replacements: dict[str, str]) -> dict:
    """Replace flagged spans.

    ``replacements`` maps a flagged claim id to the text that replaces its span. The
    engine requires an edit for *every* flagged claim, so any flagged claim without a
    replacement is filled with the empty string (deletion), which is itself a
    legitimate repair move and is exactly what the "repair deletes an accomplishment"
    negative control needs.
    """
    edits = [{"claim_id": cid, "replacement": replacements.get(cid, "")}
             for cid in flagged_claim_ids]
    return {"edits": edits}


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def restore_answer(payload: dict, replacement_for: dict[str, str]) -> dict:
    """Restate named missing facts on the lines chosen for them.

    ``replacement_for`` maps the target ``line_id`` to the new line text. The engine
    requires the answer to echo the exact text it replaces and to name the
    ``source_claim_id`` it was asked to restore.
    """
    items_by_line = {}
    for item in payload["source_claims"]:
        line = next((l for l in payload["lines"] if item["quote"].strip()
                     and l["text"].strip()), None)
        if line is not None:
            items_by_line.setdefault(line["line_id"], item)
    out = []
    for line in payload["lines"]:
        lid = line["line_id"]
        if lid not in replacement_for:
            continue
        item = items_by_line[lid]
        out.append({"line_id": lid, "source_claim_id": item["id"],
                    "before": line["text"], "replacement": replacement_for[lid]})
    return {"restorations": out}


# ---------------------------------------------------------------------------
# Convenience: the full run-7 defect, carried through to a decision
# ---------------------------------------------------------------------------

def sections_present(markdown: str) -> dict[str, bool]:
    """Which of the five promised headings survived, with or without content."""
    import re
    out = {}
    for name in ("Professional Summary", "Professional Experience", "Education",
                 "Certifications", "Technical Skills"):
        out[name] = re.search(rf"^#+\s*{re.escape(name)}\s*$", markdown,
                              re.IGNORECASE | re.MULTILINE) is not None
    return out


def section_body(markdown: str, name: str) -> str:
    """The text under a heading, for asserting it is not filler."""
    import re
    match = re.search(rf"^#+\s*{re.escape(name)}\s*\n(.*?)(?=^#+\s|\Z)", markdown,
                      re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""
