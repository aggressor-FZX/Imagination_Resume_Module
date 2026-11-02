#!/usr/bin/env python3
"""Capture actual /analyze JSON and SSE responses using synthetic offline doubles."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import pwd
import shutil
import site
import socket
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

# Resolve the account's user site-packages through the password database, not
# $HOME, because the environment below is cleared. This keeps an installed
# python-docx importable for real DOCX verification without inheriting any
# developer shell credentials or dotenv configuration.
try:
    _ACCOUNT_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
    _USER_SITE = _ACCOUNT_HOME / ".local/lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    if _USER_SITE.is_dir():
        site.addsitedir(str(_USER_SITE))
except Exception:
    _ACCOUNT_HOME = Path("/tmp")

# Nothing inherited from a developer's shell or dotenv files can become a key.
# PATH is set to fixed, non-secret system directories so pdflatex's own binary
# lookup cannot fall back to a current-directory entry.
os.environ.clear()
os.environ.update({"HOME": "/tmp", "PYTHON_DOTENV_DISABLED": "1",
                   "PATH": "/usr/local/bin:/usr/bin:/bin",
                   "PUBLICATION_RECEIPT_SECRET": "synthetic-offline-signing-key-only"})

def no_network(*args, **kwargs):
    raise AssertionError("IP network and DNS forbidden in synthetic route evidence")


def deny_ip(original):
    def checked(sock, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            return no_network()
        return original(sock, *args, **kwargs)
    return checked


# Block IP networking before importing the application and its dependencies.
# Retain AF_UNIX for in-process TestClient machinery.
socket.socket.connect = deny_ip(socket.socket.connect)
socket.socket.connect_ex = deny_ip(socket.socket.connect_ex)
socket.socket.sendto = deny_ip(socket.socket.sendto)
socket.getaddrinfo = no_network

import dotenv  # noqa: E402
from pydantic_settings.sources import DotEnvSettingsSource  # noqa: E402
dotenv.load_dotenv = lambda *args, **kwargs: False
DotEnvSettingsSource._read_env_files = lambda self: {}

from fastapi.testclient import TestClient  # noqa: E402
import app  # noqa: E402
import correction_provider  # noqa: E402
from pipeline_config import OR_SLUG_DRAFTER  # noqa: E402
from stream_context import get_token_callback  # noqa: E402

# Source, intended output, and mutations are authored independently of review
# responses. The extra blank line makes the approved output distinct from the
# source, because the real route refuses an exact source-copy fallback.
SOURCE = """## Professional Experience
**Synthetic Workshop**
- Alpha Works: Improved diagnostic accuracy by 30% over 6 months using structured test procedures, careful review of repair records, and written guidance shared with technicians during regular team meetings.
- Betaa Works: Maintained workshop inventory, coordinated equipment inspections, documented repair procedures, and kept complete maintenance records for service technicians and colleagues.

## Education
- North College: Bachelor of Mathematics, 2014, with coursework in applied statistics, quantitative reasoning, technical communication, and practical problem solving.
- South College: Master of Physics, 2016, with coursework in experimental methods, technical research, laboratory safety, and scientific reporting.

## Skills
- Diagnostics, maintenance documentation, equipment inspection, technical communication, and collaborative problem solving.
"""
# Independently authored expected output; never computed from a route response
# or from provider review decisions.
ANSWER = """## Professional Experience
**Synthetic Workshop**
- Alpha Works: Improved diagnostic accuracy by 30% over 6 months using structured test procedures, careful review of repair records, and written guidance shared with technicians during regular team meetings.
- Betaa Works: Maintained workshop inventory, coordinated equipment inspections, documented repair procedures, and kept complete maintenance records for service technicians and colleagues.


## Education
- North College: Bachelor of Mathematics, 2014, with coursework in applied statistics, quantitative reasoning, technical communication, and practical problem solving.
- South College: Master of Physics, 2016, with coursework in experimental methods, technical research, laboratory safety, and scientific reporting.

## Skills
- Diagnostics, maintenance documentation, equipment inspection, technical communication, and collaborative problem solving.
"""
ALPHA = next(line for line in SOURCE.splitlines() if line.startswith("- Alpha Works:"))
BETA = next(line for line in SOURCE.splitlines() if line.startswith("- Betaa Works:"))
WRONG = ALPHA.replace("Alpha Works:", "Betaa Works:", 1)
NORTH = next(line for line in SOURCE.splitlines() if line.startswith("- North College:"))
SOUTH = next(line for line in SOURCE.splitlines() if line.startswith("- South College:"))
SWAP_NORTH = SOUTH.replace("South College:", "North College:", 1)
SWAP_SOUTH = NORTH.replace("North College:", "South College:", 1)
EXTRA = " within 2 weeks"
INVENTED = " within 91 weeks"
SECOND = " within 92 weeks"
PHASES_REPAIR = ["source_inventory", "initial_review", "repair", "post_patch_review"]
PHASES_LIMIT = PHASES_REPAIR + ["repair", "post_patch_review"]

CASES = [
    {"name": "supported_duration_two_education", "draft": ANSWER, "mode": "honest",
     "status": "completed", "reason": None, "phases": ["source_inventory", "initial_review"]},
    {"name": "corrected_duration", "draft": ANSWER.replace(ALPHA, ALPHA + EXTRA), "mode": "repair",
     "status": "completed", "reason": None, "phases": PHASES_REPAIR},
    {"name": "wrong_employer_lying_reviewer", "draft": ANSWER.replace(ALPHA, WRONG), "mode": "wrong",
     "status": "completed", "reason": None, "phases": ["source_inventory", "initial_review"]},
    {"name": "unresolved_attribution_negative_control",
     "draft": ANSWER.replace(NORTH, SWAP_SOUTH).replace(SOUTH, SWAP_NORTH), "mode": "swap",
     "status": "needs_review", "reason": "attribution_conflict", "phases": ["source_inventory", "initial_review"]},
    {"name": "bounded_unresolved", "draft": ANSWER.replace(ALPHA, ALPHA + EXTRA), "mode": "repeat",
     "status": "needs_review", "reason": "unresolved_claims", "phases": PHASES_LIMIT},
    {"name": "post_patch_missing_fact", "draft": ANSWER.replace(ALPHA, ALPHA + EXTRA), "mode": "missing",
     "status": "needs_review", "reason": "source_facts_missing", "phases": PHASES_REPAIR},
    {"name": "post_correction_numeric_gate", "draft": ANSWER.replace(ALPHA, ALPHA + EXTRA), "mode": "numeric_lie",
     "status": "needs_review", "reason": "unsupported_claims", "phases": PHASES_REPAIR},
    {"name": "default_privacy_boundary", "draft": ANSWER, "mode": "privacy",
     "status": "needs_review", "reason": "privacy_review_required", "phases": []},
]


def inventory(payload):
    assert set(payload) == {"contract", "sources"}
    entries = []
    for source in payload["sources"]:
        start = position = 0
        for line in source["text"].splitlines(keepends=True):
            position += len(line)
            if not line.strip() or line.startswith("## "):
                continue
            quote = source["text"][start:position]
            entries.append({"id": f"{source['source_id']}:{start}:{position}",
                            "source_id": source["source_id"], "start": start,
                            "end": position, "quote": quote, "classification": "required"})
            start = position
        assert start == len(source["text"])
    return {"source_claims": entries}


def review(payload, mode, phase):
    claims = []
    represented = {item["id"]: [] for item in payload["source_claims"]}
    for line in payload["lines"]:
        text, line_id = line["text"], line["line_id"]
        if text.startswith("## "):
            claims.append({"id": f"{line_id}:0:{len(text)}", "line_id": line_id,
                           "start": 0, "end": len(text), "verdict": "structural", "citations": []})
            continue
        suffix = next((part for part in (EXTRA, INVENTED, SECOND) if text.endswith(part)), "")
        # In the numeric-gate case the post-patch reviewer lies, approving the
        # newly introduced figure as if the entire line were source-supported.
        lie_numeric = mode == "numeric_lie" and phase == "post_patch_review" and suffix == INVENTED
        base = text[:-len(suffix)] if suffix else text
        # A lying reviewer claims each swapped education record is the original.
        desired = {SWAP_NORTH: SOUTH, SWAP_SOUTH: NORTH}.get(base, base)
        item = next((item for item in payload["source_claims"]
                     if item["quote"].strip().splitlines()[-1] == desired), None)
        if item is None:
            raise ValueError(f"Unscripted synthetic line: {text!r}")
        split = len(text) if lie_numeric or not suffix else len(base)
        citation = {key: item[key] for key in ("source_id", "start", "end", "quote")}
        claims.append({"id": f"{line_id}:0:{split}", "line_id": line_id, "start": 0,
                       "end": split, "verdict": "supported", "citations": [citation]})
        represented[item["id"]].append({"line_id": line_id, "start": 0, "end": split})
        if split < len(text):
            claims.append({"id": f"{line_id}:{split}:{len(text)}", "line_id": line_id,
                           "start": split, "end": len(text), "verdict": "unsupported", "citations": []})
    mappings = []
    for item in payload["source_claims"]:
        spans = represented[item["id"]]
        if mode == "missing" and phase == "post_patch_review" and item["quote"].strip().splitlines()[-1] == NORTH:
            spans = []  # Conservative test double: the line is actually present.
        mappings.append({"source_claim_id": item["id"], "status": "represented" if spans else "missing", "spans": spans})
    return {"claims": claims, "representations": mappings}


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.content = self
        self.body = body
    async def iter_chunked(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class ScriptedTransport:
    def __init__(self, mode):
        self.mode = mode
        self.trace = []
        self.repairs = 0
        self.errors = []
        self.closed = False

    async def close(self):
        self.closed = True

    def post(self, url, **kwargs):
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        phase = ("repair" if "flagged_claim_ids" in payload else
                 "source_inventory" if "lines" not in payload else
                 "initial_review" if not any(t["phase"] == "initial_review" for t in self.trace)
                 else "post_patch_review")
        record = {"phase": phase, "payload": copy.deepcopy(payload),
                  "request_policy": {"allow_redirects": kwargs.get("allow_redirects"),
                                     "provider": kwargs["json"]["provider"]}}
        self.trace.append(record)
        try:
            if phase == "source_inventory":
                body = inventory(payload)
            elif phase == "repair":
                self.repairs += 1
                replacement = (INVENTED if self.repairs == 1 else SECOND) if self.mode == "repeat" else (
                    INVENTED if self.mode == "numeric_lie" else "")
                body = {"edits": [{"claim_id": cid, "replacement": replacement}
                                  for cid in payload["flagged_claim_ids"]]}
            else:
                body = review(payload, self.mode, phase)
            record["response"] = copy.deepcopy(body)
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            record["script_error"] = self.errors[-1]
            raise
        envelope = {"model": OR_SLUG_DRAFTER,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}]}
        return FakeResponse(json.dumps(envelope).encode())


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        events.append({"event": next(line[7:] for line in lines if line.startswith("event: ")),
                       "data": json.loads(next(line[6:] for line in lines if line.startswith("data: ")))})
    return events


def _python_docx_version():
    try:
        import docx
        return getattr(docx, "__version__", "installed")
    except Exception:
        return None


def export_texts(client, result):
    """Call the real /resume/export route and extract actual PDF and DOCX text.

    Only text and byte counts are stored in the evidence bundle; the binary
    files and temporary paths are discarded.
    """
    import base64
    import io
    import shutil
    import subprocess
    import tempfile
    from docx import Document

    headers = {"X-Cogito-Owner-Id": "synthetic-offline-owner"}
    body = {**result["approved_export_payload"],
            "final_written_section_markdown": result["final_written_section_markdown"],
            "publication_receipt": result["publication_receipt"],
            "metrics_support": result.get("metrics_support")}
    response = client.post("/resume/export", json=body, headers=headers)
    exported = {"status": response.status_code, "pdf_text": None, "docx_text": None,
                "pdf_bytes": None, "docx_bytes": None, "filename": None,
                "pdf_extractor": None, "reason": None}
    if response.status_code != 200:
        exported["reason"] = (response.json() or {}).get("reasons")
        return exported
    payload = response.json()
    exported["filename"] = payload.get("filename")
    pdf_bytes = base64.b64decode(payload["pdf_base64"])
    docx_bytes = base64.b64decode(payload["docx_base64"])
    exported["pdf_bytes"], exported["docx_bytes"] = len(pdf_bytes), len(docx_bytes)
    document = Document(io.BytesIO(docx_bytes))
    exported["docx_text"] = "\n".join(paragraph.text for paragraph in document.paragraphs)
    if shutil.which("pdftotext"):
        with tempfile.TemporaryDirectory(prefix="correction-export-check-") as directory:
            pdf = Path(directory) / "approved.pdf"
            pdf.write_bytes(pdf_bytes)
            completed = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                                       capture_output=True, check=False)
            if completed.returncode == 0:
                exported["pdf_extractor"] = "pdftotext -layout"
                exported["pdf_text"] = completed.stdout.decode("utf-8", "replace")
    return exported


def execute(client, spec, format_name):
    transport = ScriptedTransport(spec["mode"])
    app.app.state.http_session = transport
    observations = []
    original_correct = app.correct

    async def observe(*args, **kwargs):
        result = await original_correct(*args, **kwargs)
        observations.append(dataclasses.asdict(result))
        return result

    async def pipeline(**kwargs):
        callback = get_token_callback()
        if callback:
            callback("rewrite", "PRIVATE_RAW_STREAM_TOKEN")
        return {"pipeline_status": "completed", "final_written_section_markdown": spec["draft"],
                "rewritten_resume": "PRIVATE_STALE_DRAFT", "final_written_section": "PRIVATE_STALE_DRAFT",
                "run_metrics": {"calls": [], "failures": [], "total_prompt_tokens": 0,
                                "total_completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0}}

    headers = {"X-Cogito-Owner-Id": "synthetic-offline-owner"}
    if format_name == "sse":
        headers["Accept"] = "text/event-stream"
    body = {"resume_text": SOURCE, "job_ad": "Synthetic equipment maintenance role",
            "source_documents": [{"source_id": "candidate-one", "text": SOURCE, "kind": "candidate_document"}]}
    disclosure = (correction_provider._approved_disclosure_available if spec["mode"] == "privacy"
                  else lambda: True)
    with patch.object(app, "correct", observe), patch.object(app, "run_new_pipeline_async", pipeline), \
         patch.object(correction_provider, "_approved_disclosure_available", disclosure):
        response = client.post("/analyze", json=body, headers=headers)
    events = parse_sse(response.text) if format_name == "sse" else []
    result = next((event["data"] for event in events if event["event"] == "result"), {}) if events else response.json()
    checks = []

    def check(name, actual, expected):
        checks.append({"assertion": name, "expected": expected, "actual": actual, "pass": actual == expected})

    check("http_status", response.status_code, 200)
    check("content_type", response.headers.get("content-type", "").split(";")[0],
          "text/event-stream" if format_name == "sse" else "application/json")
    check("script_errors", transport.errors, [])
    check("correction_observations", len(observations), 1)
    check("provider_phases", [entry["phase"] for entry in transport.trace], spec["phases"])
    check("reported_phases", [call["phase"] for call in (result.get("run_metrics") or {}).get("calls", [])
                              if call.get("stage") == "source_correction"], spec["phases"])
    check("processing_status", result.get("processing_status"), spec["status"])
    check("correction_reason", result.get("correction_reason"), spec["reason"])
    check("correction_status", result.get("correction_status"),
          "approved" if spec["status"] == "completed" else "withheld")
    check("review_reason", result.get("review_reason"), spec["reason"])
    check("raw_stream_and_stale_draft_not_published", "PRIVATE_" not in response.text, True)
    approved = spec["status"] == "completed"
    check("requires_review", result.get("requires_review"), not approved)
    for field in ("final_written_section_markdown", "final_written_section", "rewritten_resume"):
        check(field, result.get(field), ANSWER if approved else None)
    check("receipt_present", bool(result.get("publication_receipt")), approved)
    check("export_payload_present", bool(result.get("approved_export_payload")), approved)
    if approved:
        check("numeric_verifier", (result.get("metrics_support") or {}).get("verdict"), "PASS (tier A only)")
        check("education_records_in_export", (result.get("approved_export_payload") or {}).get("education"),
              [NORTH.removeprefix("- "), SOUTH.removeprefix("- ")])
    else:
        check("rejected_draft_not_published", spec["draft"] not in response.text, True)
    if spec["mode"] in {"repair", "missing"}:
        post = [entry for entry in transport.trace if entry["phase"] == "post_patch_review"]
        post_lines = [line["text"] for line in post[0]["payload"]["lines"]] if post else []
        check("post_patch_review_has_answer_lines", post_lines,
              [line for line in ANSWER.splitlines() if line.strip()])
    if spec["mode"] == "wrong":
        check("wrong_employer_restored_single_owner",
              "restored 1 source-owned record label" in (result.get("correction_commentary") or "").lower(),
              True)
        check("wrong_employer_not_in_approved_content",
              WRONG not in (result.get("final_written_section_markdown") or ""), True)
    if spec["mode"] == "swap":
        check("actual_attribution_decision", observations[0]["reason"] if observations else None,
              "attribution_conflict")
        check("no_callback_or_schema_failure", observations[0]["reason"] not in {
            "provider_error", "malformed_output", "timeout"} if observations else None, True)
    if spec["mode"] == "numeric_lie":
        check("correction_approved_before_numeric_gate", observations[0]["status"] if observations else None, "approved")
        check("post_correction_numeric_input", observations[0]["markdown"] if observations else None,
              ANSWER.replace(ALPHA, ALPHA + INVENTED))
    if format_name == "sse":
        check("sse_event_sequence", [event["event"] for event in events],
              ["token", "result"] if approved else ["withheld", "result"])
        if approved:
            check("sse_token", events[0]["data"] if events else None, {"field": "rewrite", "delta": ANSWER})
    export = None
    if approved:
        export = export_texts(client, result)
        check("export_http_status", export["status"], 200)
        check("export_has_pdf_and_docx", (export["pdf_bytes"] or 0) > 0
              and (export["docx_bytes"] or 0) > 0, True)
        check("pdf_text_extracted", export.get("pdf_extractor"), "pdftotext -layout")
        education_line = NORTH.split(": ", 1)[1].strip()
        for label, text in (("pdf", export["pdf_text"] or ""),
                            ("docx", export["docx_text"] or "")):
            flat = " ".join(text.split())
            for fact in (ALPHA.split(": ", 1)[1].strip(),
                         BETA.split(": ", 1)[1].strip(),
                         education_line,
                         SOUTH.split(": ", 1)[1].strip()):
                check(f"{label}_contains_approved_fact", fact in flat, True)
            check(f"{label}_excludes_wrong_attribution_pairing",
                  "Betaa Works: Improved diagnostic accuracy" not in flat, True)
            check(f"{label}_excludes_invented_duration", EXTRA.strip() not in flat, True)
    return {"name": spec["name"], "format": format_name, "request": body,
            "draft": spec["draft"], "expectations": {k: spec[k] for k in ("status", "reason", "phases")},
            "http": {"status": response.status_code, "content_type": response.headers.get("content-type"),
                     "raw_body": response.text, "parsed_result": result, "sse_events": events},
            "export": export,
            "correction_observations": observations, "provider_trace": transport.trace, "checks": checks}


def main():
    app.settings.openrouter_api_key_1 = "synthetic-key"
    app.settings.openrouter_api_key_2 = None
    app.app.dependency_overrides[app.get_api_key] = lambda: "synthetic-auth"
    with TestClient(app.app) as client:
        client.portal.call(app.app.state.http_session.close)
        runs = [execute(client, spec, fmt) for spec in CASES for fmt in ("json", "sse")]
    app.app.dependency_overrides.clear()
    failed = sum(not check["pass"] for run in runs for check in run["checks"])
    sources = ["app.py", "source_correction.py", "correction_provider.py", "publication_gate.py",
               "tier_a_verifier.py", "approved_resume.py"]
    bundle = {"notice": "Synthetic only. Actual in-process route responses; scripted pipeline/reviewer/transport, no network or dotenv. Not model or production proof.",
              "working_tree_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
              "answer_key": {"source": SOURCE, "approved_markdown": ANSWER,
                             "achievement": ALPHA, "education_records": [NORTH, SOUTH]},
              "render_dependencies": {
                  "pdflatex": shutil.which("pdflatex"),
                  "pdftotext": shutil.which("pdftotext"),
                  "python_docx": _python_docx_version(),
              },
              "runs": runs, "summary": {"cases": len(CASES), "route_requests": len(runs),
                                          "assertions": sum(len(run["checks"]) for run in runs),
                                          "failed_assertions": failed}}
    output = HERE / "route_results.json"
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for run in runs:
        failures = [check["assertion"] for check in run["checks"] if not check["pass"]]
        print(f"{run['name']} / {run['format']}: " + ("UNMET " + ", ".join(failures) if failures else "matched"))
    print(f"Wrote {output}; {failed} unmet assertions")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())