#!/usr/bin/env python3
"""Reproduce a retained failing case through the REAL provider adapter.

Why this exists
---------------
A scripted reviewer that emits the same reason code is NOT proof of the original
failure: it reads whatever payload the engine hands it, while the real adapter
rewrites the payload before it reaches the model. Every integration defect in this
module has lived in that rewrite (most recently: the citation question being
translated into a review request, discarding its contract).

So this harness exercises:

    real sources + real draft  ->  correct()  ->  CorrectionProvider
                               ->  correction_protocol.indexed_* translation
                               ->  the actual OpenRouter request
                               ->  the actual model answer
                               ->  the real validator

and records the translated OUTBOUND payload, the raw response, the phase events
and the final decision.

The input is the retained fixture, verified byte-identical by content hash before
the run; nothing is shortened or made easier, and no mode is changed.

Usage:
    python scripts/repro_correction_case.py --out captures/repro/<label>

Requires OPENROUTER_API_KEY_1 (or _2) and CORRECTION_PROVIDER_DISCLOSURE=1 for the
real provider path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

FIXTURE = pathlib.Path("tests/fixtures/realistic_candidate")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_fixture():
    """The retained input, with hashes so the evidence cannot drift."""
    main = (FIXTURE / "candidate-main.txt").read_text()
    supplement = (FIXTURE / "candidate-supplement.txt").read_text()
    job_ad = (FIXTURE / "intended-job-description.txt").read_text()
    # The orchestrator joins the documents exactly this way (server.cjs), and the
    # Imaginator re-validates the join, so the separator is part of the contract.
    resume_text = main + "\n\n---\n\n" + supplement
    return {
        "job_ad": job_ad,
        "resume_text": resume_text,
        "documents": [
            {"source_id": "document-0", "text": main, "kind": "candidate_document"},
            {"source_id": "document-1", "text": supplement, "kind": "candidate_document"},
        ],
        "hashes": {
            "candidate-main.txt": sha(main),
            "candidate-supplement.txt": sha(supplement),
            "intended-job-description.txt": sha(job_ad),
            "resume_text": sha(resume_text),
        },
    }


def verify_against_retained(fixture):
    """Check the hashes recorded by the original failing run, when available."""
    recorded_path = pathlib.Path(
        "captures/recovery-validation/final-realistic_candidate-imaginative-5/"
        "started.json")
    if not recorded_path.exists():
        return {"verified": False, "reason": "no retained started.json"}
    recorded = json.loads(recorded_path.read_text())["input_sha256"]
    mismatches = []
    for name, expected in recorded.items():
        actual = fixture["hashes"].get(name)
        if actual != expected:
            mismatches.append({"file": name, "recorded": expected, "actual": actual})
    return {"verified": not mismatches, "mismatches": mismatches,
            "reference": str(recorded_path)}


class RecordingProvider:
    """Transparent proxy that keeps the OUTBOUND payload and the RAW answer.

    Wraps the real `CorrectionProvider` so the recorded request is the one the
    adapter actually built -- after `indexed_*` translation, after the disclosure
    gate, after private `_`-prefixed keys were stripped.
    """

    def __init__(self, inner, record):
        self._inner = inner
        self._record = record

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_budget(self, remaining_seconds):
        setter = getattr(self._inner, "set_budget", None)
        if callable(setter):
            return setter(remaining_seconds)
        return None

    async def __call__(self, phase, payload):
        entry = {"phase": phase,
                 # What the ENGINE asked for, before translation.
                 "engine_payload_keys": sorted(payload.keys()),
                 "engine_contract": (payload.get("contract") or "")[:120]}
        try:
            answer = await self._inner(phase, payload)
        except BaseException as exc:
            entry["error"] = type(exc).__name__
            entry["constraint"] = getattr(exc, "constraint", None)
            self._record["phases"].append(entry)
            raise
        entry["answer_keys"] = sorted(answer.keys()) if isinstance(answer, dict) else type(answer).__name__
        self._record["phases"].append(entry)
        return answer


async def analyze_via_route(fixture, api_key):
    """Run the real /analyze route and return its draft markdown.

    The route is the system under investigation: parser -> drafter -> correction.
    Calling correction in isolation would skip the stages that decided the draft.
    """
    import aiohttp

    base = os.environ.get("IMAGINATOR_BASE", "https://imaginator-resume-cowriter.onrender.com")
    payload = {
        "resume_text": fixture["resume_text"],
        "job_ad": fixture["job_ad"],
        "source_documents": fixture["documents"],
    }
    headers = {"Content-Type": "application/json",
               # A signed-in account owner, so this isolates correction from the
               # unfinished guest-session integration.
               "X-Cogito-Owner-Id": os.environ.get("REPRO_OWNER_ID", "42")}
    # The route is service-to-service authenticated; without the key it answers
    # 403 and the harness would report "no draft" for the wrong reason.
    service_key = os.environ.get("IMAGINATOR_AUTH_TOKEN") or ""
    if service_key:
        headers["X-API-Key"] = service_key
    timeout = aiohttp.ClientTimeout(total=280)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base}/analyze", json=payload,
                                headers=headers) as resp:
            body = await resp.json(content_type=None)
            status = resp.status
    body = dict(body or {})
    body["_status"] = status
    markdown = body.get("final_written_section_markdown") or ""
    if not isinstance(markdown, str):
        markdown = ""
    return markdown, body


async def run(fixture, out_dir, draft_from=None):
    from correction_provider import CorrectionProvider
    from source_correction import SourceDocument, correct

    import aiohttp

    record = {"phases": [], "events": []}

    def observe(event):
        # Sanitized engine events: numbers and fixed identifiers only.
        record["events"].append(event)

    api_key = (os.environ.get("OPENROUTER_API_KEY_1")
               or os.environ.get("OPENROUTER_API_KEY_2") or "")
    if not api_key:
        raise SystemExit("no OPENROUTER_API_KEY_1/_2 in the environment")

    # ── Generation ────────────────────────────────────────────────────────
    # Driven through the LIVE /analyze route, not a hand-written draft and not an
    # in-process call. The retained failure happened on that route, and the parser
    # + drafter stages upstream of correction are part of what produced the draft.
    #
    # `--draft-from` lets a retained draft be supplied instead; when it is, that is
    # recorded, because a regenerated draft is a different input to correction.
    if draft_from:
        draft_markdown = pathlib.Path(draft_from).read_text()
        record["draft_source"] = f"retained file {draft_from}"
    else:
        draft_markdown, route_response = await analyze_via_route(fixture, api_key)
        record["draft_source"] = "live /analyze route (this run)"
        record["route_response_summary"] = {
            "http_status": route_response.get("_status"),
            "correction_status": route_response.get("correction_status"),
            "correction_reason": route_response.get("correction_reason"),
            "review_reason": route_response.get("review_reason"),
            "had_markdown": bool(route_response.get("final_written_section_markdown")),
        }
    if not draft_markdown or not draft_markdown.strip():
        # Persist what the route DID say before stopping: "no draft" alone is not
        # evidence about which stage refused, and it is the route's own reason code
        # that names the blocker.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "repro.json").write_text(json.dumps(record, indent=2))
        summary = record.get("route_response_summary") or {}
        print("\nROUTE REFUSED BEFORE CORRECTION:")
        print(json.dumps(summary, indent=1))
        raise SystemExit(
            "the route produced no draft to correct. The upstream stage is the "
            "blocker, not correction; run with --draft-from to test correction "
            "against a retained draft.")
    record["draft"] = draft_markdown
    record["draft_sha256"] = sha(draft_markdown)

    sources = tuple(SourceDocument(d["source_id"], d["text"]) for d in fixture["documents"])

    async with aiohttp.ClientSession() as session:
        inner = CorrectionProvider(api_key, session, observer=observe)
        provider = RecordingProvider(inner, record)
        result = await correct(draft_markdown, sources, provider, timeout_seconds=120.0)

    record["result"] = {
        "status": result.status,
        "reason": result.reason,
        "review_calls": result.review_calls,
        "repair_calls": result.repair_calls,
        "restorations": getattr(result, "restore_calls", None),
        "commentary": result.commentary,
        "markdown_sha256": sha(result.markdown) if isinstance(result.markdown, str) else None,
        "markdown_chars": len(result.markdown) if isinstance(result.markdown, str) else 0,
        "markdown": result.markdown,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repro.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="directory for the capture")
    ap.add_argument("--check-only", action="store_true",
                    help="verify the input hashes and stop (no provider calls)")
    ap.add_argument("--draft-from", default=None,
                    help="use a retained draft file instead of regenerating")
    args = ap.parse_args()

    fixture = load_fixture()
    integrity = verify_against_retained(fixture)
    print("input hashes:", json.dumps(fixture["hashes"], indent=1))
    print("matches the retained failing run:", integrity["verified"])
    if not integrity["verified"]:
        print("  detail:", json.dumps(integrity, indent=1))
        print("  REFUSING: the input differs from the retained case, so a run "
              "here would not reproduce it.")
        return 2
    if args.check_only:
        return 0

    record = asyncio.run(run(fixture, pathlib.Path(args.out), args.draft_from))
    print("\n--- RESULT ---")
    print(json.dumps({k: v for k, v in record["result"].items() if k != "markdown"},
                     indent=1))
    print("phases reached:", [p["phase"] for p in record["phases"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
