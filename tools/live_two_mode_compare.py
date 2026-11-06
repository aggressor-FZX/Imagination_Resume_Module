#!/usr/bin/env python3
"""Live two-mode comparison runner for the bridging fixture.

Runs the SAME candidate through the real orchestrator multi-file route in both
modes, exports the delivered files, and writes everything it observed to disk so
the comparison is made from artefacts rather than from what a run "should" have
done.

Usage:
    python3 tools/live_two_mode_compare.py <mode>
        mode is `imaginative` or `conservative`

Reads the session cookie from /tmp/bridge_cookie.txt so the account is created
once, outside this script.
"""
from __future__ import annotations

import base64
import json
import pathlib
import secrets
import sys
import urllib.error
import urllib.request

BASE = "https://orchestrator-backend-isj7.onrender.com"
FIXTURE = pathlib.Path(
    "/home/skystarved/Render_Dockers/Imagination_Resume_Module/"
    ".worktrees/resumeDraft_fix/tests/fixtures/bridge_opportunity")
OUT = pathlib.Path("/tmp/bridge_run")


def _post_json(path, payload, cookie, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Cookie": cookie})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _multipart(fields, files):
    boundary = "----B" + secrets.token_hex(8)
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f'\r\n\r\n{value}\r\n'.encode())
    for path in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
            f'filename="{path.name}"\r\nContent-Type: application/vnd.'
            f'openxmlformats-officedocument.wordprocessingml.document'
            f'\r\n\r\n'.encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def analyze(mode, cookie):
    fields = {
        "job_title": "Operations Analyst",
        "job_ad": (FIXTURE / "intended-job-description.txt").read_text(),
        "job_location": "Tacoma, WA",
        # Style deliberately held constant so the ONLY variable is inference.
        "creativity_mode": "balanced",
        "skill_inference_mode": mode,
    }
    body, ctype = _multipart(
        fields, [FIXTURE / "candidate-main.docx",
                 FIXTURE / "candidate-supplement.docx"])
    req = urllib.request.Request(
        BASE + "/api/analysis/multi-file?stream=0", data=body, method="POST",
        headers={"Content-Type": ctype, "Cookie": cookie,
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def export(analysis, cookie):
    """Export exactly what buildExportPayload() sends."""
    detail = analysis.get("imaginationDetails") or {}
    canonical = analysis.get("approved_export_payload") or {}
    payload = {
        **canonical,
        "approved_export_payload": canonical,
        "final_written_section_markdown":
            analysis.get("final_written_section_markdown"),
        "publication_receipt": analysis.get("publication_receipt"),
        "skill_inference_policy_state":
            detail.get("skill_inference_policy_state"),
    }
    return _post_json("/api/ml/imagination/export", payload, cookie)


def main():
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    cookie = pathlib.Path("/tmp/bridge_cookie.txt").read_text().strip()

    status, raw = analyze(mode, cookie)
    try:
        analysis = json.loads(raw)
    except Exception:
        print(f"{mode}: non-JSON response {status}: {raw[:300]}")
        return 1
    (OUT / f"{mode}_analysis.json").write_text(json.dumps(analysis, indent=1))

    detail = analysis.get("imaginationDetails") or {}
    policy = detail.get("skill_inference_policy_state") or {}
    md = analysis.get("final_written_section_markdown") or ""
    print(f"[{mode}] analyze status={status} "
          f"review={analysis.get('requires_review')} "
          f"reasons={analysis.get('review_reasons')}")
    print(f"   reported_mode={detail.get('skill_inference_mode')} "
          f"policy={detail.get('skill_inference_policy_version')}")
    print(f"   surviving_bridges={json.dumps(policy.get('surviving_bridges'))}")
    print(f"   words={len(md.split())} receipt={bool(analysis.get('publication_receipt'))}")

    if not analysis.get("publication_receipt"):
        print(f"   -> no receipt, nothing to export")
        return 0

    status, raw = export(analysis, cookie)
    try:
        exported = json.loads(raw)
    except Exception:
        print(f"   export non-JSON {status}: {raw[:200]}")
        return 1
    print(f"   export status={status}")
    if status != 200:
        print(f"   export refused: {exported.get('reasons') or exported.get('error')}")
        return 0

    pdf = base64.b64decode(exported["pdf_base64"])
    docx = base64.b64decode(exported["docx_base64"])
    (OUT / f"{mode}.pdf").write_bytes(pdf)
    (OUT / f"{mode}.docx").write_bytes(docx)
    print(f"   filename={exported.get('filename')!r} "
          f"pdf={len(pdf)}B docx={len(docx)}B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
