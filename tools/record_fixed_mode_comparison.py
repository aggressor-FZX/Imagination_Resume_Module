#!/usr/bin/env python3
"""One attempt per mode; record every response without overwriting evidence.

Uses an already-authenticated synthetic session in /tmp/bridge_cookie.txt.
Never registers an account or retries; run only with explicit paid-run approval.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid

import live_two_mode_compare as transport


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "captures" / "restoration-mapping-validation"
EXPECTED = {
    "candidate-main.docx": "7883ee26ab0dc57f8493be8856317d84e3a5c9bed3a48f123715347c9ef7a2b9",
    "candidate-supplement.docx": "d2263b6311f501da6e925ab401cb279ab4ed2d72ec53b666f11a40548bacd8b9",
    "candidate-main.txt": "d3c2a8e0634d3035b5dd3767d7d8469b7e2934efaf325901def3823106988732",
    "candidate-supplement.txt": "5e1e4b31a492eda862f3d15915b563717b0b44a5a50775c3897dc9531fbd6284",
    "intended-job-description.txt": "88de858c90a31e4f4955c23ad31d059179b8e47ff6caa374577a4cd348ef26d1",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    with path.open("xb") as target:
        target.write(data)


def save_json(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False).encode())


def run(mode):
    if mode not in {"imaginative", "conservative"}:
        raise ValueError("unsupported mode")
    for name, expected in EXPECTED.items():
        if digest((transport.FIXTURE / name).read_bytes()) != expected:
            raise RuntimeError("frozen input mismatch: " + name)
    cookie = Path("/tmp/bridge_cookie.txt").read_text().strip()
    req = urllib.request.Request(transport.BASE + "/api/auth/me", headers={"Cookie": cookie})
    with urllib.request.urlopen(req, timeout=30) as response:
        if not json.load(response).get("isAuthenticated"):
            raise RuntimeError("synthetic session is not authenticated")

    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(OUT, 0o700)
    attempt = OUT / (mode + "-" + uuid.uuid4().hex)
    attempt.mkdir(mode=0o700)
    summary = {"mode": mode, "started_unix": time.time(), "input_sha256": EXPECTED,
               "status": "started", "cost_scope": "reported Imaginator usage only"}
    save_json(attempt / "started.json", summary)
    try:
        status, raw = transport.analyze(mode, cookie)
        write(attempt / "analysis.raw", raw.encode())
        data = json.loads(raw)
        detail = data.get("imaginationDetails") or {}
        metrics = detail.get("run_metrics") or {}
        enrichment = (data.get("enrichment") or {}).get("details") or {}
        summary.update(http_status=status, run_id=data.get("runId"),
                       review_reasons=data.get("review_reasons"),
                       requires_review=data.get("requires_review"),
                       reported_mode=detail.get("skill_inference_mode"),
                       receipt_issued=bool(data.get("publication_receipt")),
                       model_cost_usd=metrics.get("estimated_cost_usd"),
                       model_calls=len(metrics.get("calls") or []),
                       enrichment_paid_calls=enrichment.get("paid_calls"),
                       enrichment_credits=enrichment.get("credits"),
                       enrichment_unknown_outcomes=enrichment.get("unknown_outcomes"),
                       status="withheld" if data.get("requires_review") else "response_received")
        if data.get("publication_receipt") and not data.get("requires_review"):
            export_status, export_raw = transport.export(data, cookie)
            write(attempt / "export.raw", export_raw.encode())
            summary["export_http_status"] = export_status
            if export_status == 200:
                exported = json.loads(export_raw)
                for kind in ("pdf", "docx"):
                    content = base64.b64decode(exported[kind + "_base64"], validate=True)
                    if not content:
                        raise ValueError("empty " + kind)
                    write(attempt / (mode + "." + kind), content)
                    summary[kind + "_sha256"] = digest(content)
                summary["status"] = "exported"
            else:
                summary["status"] = "export_refused"
    except Exception as exc:
        summary["status"] = "error"
        summary["error_type"] = type(exc).__name__
    finally:
        summary["finished_unix"] = time.time()
        save_json(attempt / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    print("private_record:", attempt)
    return 0 if summary["status"] in {"withheld", "exported"} else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))