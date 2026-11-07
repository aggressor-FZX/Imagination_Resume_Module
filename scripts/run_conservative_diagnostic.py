#!/usr/bin/env python3
"""One authorized Conservative diagnostic against the frozen fixture.

This is a *diagnostic*, not a benchmark: it sends the frozen bridge-opportunity
input to the live Imaginator's ``/analyze`` route once, records the full response
verbatim, and prints only structural facts (statuses, reason codes, call phases,
counts). It never retries, so a withheld result is preserved as evidence rather
than papered over.

The purpose is to answer one bounded question: after the ``_recover_unjudged``
fix, does a post-repair omission get completed instead of withheld?

Usage::

    IMAGINATOR_AUTH_TOKEN=... python scripts/run_conservative_diagnostic.py \
        --label post-recovery-fix

Credentials come from the environment and are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import secrets
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = HERE / "tests" / "fixtures" / os.environ.get("DIAG_FIXTURE", "bridge_opportunity")
ENDPOINT = "https://imaginator-resume-cowriter.onrender.com/analyze"
INPUT_FILES = ("candidate-main.txt", "candidate-supplement.txt",
               "intended-job-description.txt")


def _read(name: str) -> tuple[str, str]:
    """Return ``(text, sha256)`` for one frozen fixture file."""
    raw = (FIXTURE / name).read_bytes()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _post(payload: dict, token: str, timeout: float) -> tuple[int, dict | str]:
    # The direct route has no authenticated session, so the diagnostic supplies
    # the trusted owner header the orchestrator would normally derive from it.
    # Without it the approval cannot be signed and the run withholds as
    # `receipt_unavailable` even when the correction itself approved (measured:
    # attempt 3, diagcons-2cfe7698eb37). This is a harness concern, not a
    # product change: the header is never taken from the request body.
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": token,
                 "X-Cogito-Owner-Id": "diagnostic-owner"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - diagnostics must record, not raise
        return 0, f"{type(exc).__name__}: {exc}"


def _phases(body: dict) -> list[str]:
    calls = ((body.get("run_metrics") or {}).get("calls")) or []
    return [f"{call.get('stage')}/{call.get('phase')}" for call in calls
            if call.get("stage") == "source_correction"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="record directory name")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--mode", choices=("conservative", "imaginative"),
                        default="conservative")
    args = parser.parse_args(argv)

    token = os.environ.get("IMAGINATOR_AUTH_TOKEN", "").strip()
    if not token:
        print("IMAGINATOR_AUTH_TOKEN is not set", file=sys.stderr)
        return 2

    correlation_id = f"diagcons-{secrets.token_hex(6)}"
    main_text, main_sha = _read("candidate-main.txt")
    supp_text, supp_sha = _read("candidate-supplement.txt")
    job_text, job_sha = _read("intended-job-description.txt")

    # The route checks `resume_text` against its source documents exactly:
    # `_snapshot_original_documents` returns None unless the texts join to it, and
    # a None snapshot withholds as `source_documents_mismatch`. So the resume text
    # is derived from the documents rather than read from a third file.
    documents = [
        {"source_id": "candidate-main.txt", "text": main_text,
         "kind": "candidate_document"},
        {"source_id": "candidate-supplement.txt", "text": supp_text,
         "kind": "candidate_document"},
    ]
    payload = {
        "correlation_id": correlation_id,
        "skill_inference_mode": args.mode,
        "resume_text": "\n\n---\n\n".join(doc["text"] for doc in documents),
        "source_documents": documents,
        "job_ad": job_text,
    }

    record = HERE / "captures" / "recovery-validation" / args.label
    record.mkdir(parents=True, exist_ok=True)
    (record / "started.json").write_text(json.dumps({
        "mode": args.mode,
        "route": "imaginator /analyze direct",
        "correlation_id": correlation_id,
        "input_sha256": {
            "candidate-main.txt": main_sha,
            "candidate-supplement.txt": supp_sha,
            "intended-job-description.txt": job_sha,
        },
        "job_search_calls": 0,
        "started_unix": time.time(),
    }, indent=1) + "\n", encoding="utf-8")

    started = time.time()
    status, body = _post(payload, token, args.timeout)
    elapsed = time.time() - started

    if isinstance(body, str):
        (record / "response.raw").write_text(body, encoding="utf-8")
        print(f"HTTP {status} after {elapsed:.1f}s -- see response.raw")
        return 1

    (record / "response.json").write_text(
        json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "correlation_id": correlation_id,
        "http_status": status,
        "elapsed_seconds": round(elapsed, 1),
        "correction_status": body.get("correction_status"),
        "correction_reason": body.get("correction_reason"),
        "review_reasons": body.get("review_reasons"),
        "reported_mode": body.get("skill_inference_mode"),
        "policy_version": body.get("skill_inference_policy_version"),
        "processing_status": body.get("processing_status"),
        "phases": _phases(body),
        "restore_calls": body.get("restore_calls"),
        "verification_coverage": body.get("verification_coverage"),
        "skill_bridges": body.get("skill_bridges"),
        "estimated_cost_usd": (body.get("run_metrics") or {}).get("estimated_cost_usd"),
    }
    (record / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
