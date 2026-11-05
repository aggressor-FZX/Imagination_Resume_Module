#!/usr/bin/env python3
"""Live acceptance probes for the two-mode skill bridging feature.

Reads configuration from the environment — never hardcodes a key. Two modes are
checked against the deployed service; the assertions are about *behaviour a
customer would see*, not internal plumbing.

Usage:
    IMAGINATOR_URL=... IMAGINATOR_AUTH_TOKEN=... python3 live_bridge_probe.py

Exit code 0 only when every probe passes. Each probe prints what it observed so a
failure is diagnosable without re-running blindly.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


BASE = os.environ.get(
    "IMAGINATOR_URL", "https://imaginator-resume-cowriter.onrender.com"
).rstrip("/")
TOKEN = os.environ.get("IMAGINATOR_AUTH_TOKEN", "")

# Deliberately synthetic: fabricated person, fabricated employer, no real PII.
SOURCE = (
    "Jordan Example\n"
    "Cascade Power Solutions\n"
    "Compared weekly sales figures to identify trends across 12 branches.\n"
    "Trained 6 apprentices on lockout/tagout and NFPA 70E.\n"
)
JOB_AD = (
    "Senior Data Analyst. Requires SQL, Python, and trend analysis experience. "
    "Must be able to build forecasting models."
)

DRAFT = (
    "## Professional Summary\n"
    "Analyst with measurement and reporting experience.\n\n"
    "## Professional Experience\n"
    "**Cascade Power Solutions** -- Example City, WA\n"
    "*Data Analyst | 03/2020 -- Present*\n"
    "- Compared weekly sales figures to identify trends across 12 branches.\n"
    "- Trained 6 apprentices on lockout/tagout and NFPA 70E.\n\n"
    "## Education\n"
    "**Example Technical College** -- 2016\n\n"
    "## Certifications\n\n"
    "## Technical Skills\n"
    "- trend analysis\n"
)

# ── Receipt-capable fixture ────────────────────────────────────────────────
# The fixture above is deliberately small, and the publication gate requires 80
# words with all required sections present, so it can never reach an approved
# result — which is why the receipt probe kept skipping. This pair is written to
# clear every *structural* gate so a live run has a chance of producing a v5
# receipt. It cannot guarantee approval: whether the live reviewer supports each
# line is a semantic judgement this probe must not pre-empt.
#
# Everything here is synthetic: fabricated person, employer, college and figures.
FULL_SOURCE = (
    "Jordan Example\n"
    "Example City, WA | jordan.example@example.test\n\n"
    "Cascade Power Solutions - Data Analyst - 03/2020 to Present\n"
    "Compared weekly sales figures across 12 branches to identify trends.\n"
    "Built the regional reporting pack used by four operations managers.\n"
    "Trained 6 apprentices on lockout/tagout and NFPA 70E procedures.\n\n"
    "Northline Logistics - Reporting Assistant - 06/2017 to 02/2020\n"
    "Reconciled monthly freight invoices against carrier statements.\n"
    "Maintained the shipment tracking workbook for the western region.\n\n"
    "Example Technical College - Associate of Applied Science - 2016\n"
    "Coursework included statistics, spreadsheet modelling and technical writing.\n"
)

FULL_DRAFT = (
    "## Professional Summary\n"
    "Data analyst with five years of measurement and reporting experience across "
    "field operations and logistics. Builds recurring reporting that operations "
    "managers use to compare branch performance and to track shipment activity. "
    "Comfortable reconciling large invoice sets and maintaining the workbooks "
    "that other teams depend on for their weekly decisions.\n\n"
    "## Professional Experience\n"
    "**Cascade Power Solutions** -- Example City, WA\n"
    "*Data Analyst | 03/2020 -- Present*\n"
    "- Compared weekly sales figures across 12 branches to identify trends.\n"
    "- Built the regional reporting pack used by four operations managers.\n"
    "- Trained 6 apprentices on lockout/tagout and NFPA 70E procedures.\n\n"
    "**Northline Logistics** -- Example City, WA\n"
    "*Reporting Assistant | 06/2017 -- 02/2020*\n"
    "- Reconciled monthly freight invoices against carrier statements.\n"
    "- Maintained the shipment tracking workbook for the western region.\n\n"
    "## Education\n"
    "**Example Technical College** -- 2016\n"
    "Associate of Applied Science. Coursework included statistics, spreadsheet "
    "modelling and technical writing.\n\n"
    "## Certifications\n\n"
    "## Technical Skills\n"
    "- trend analysis\n"
    "- reporting\n"
    "- invoice reconciliation\n"
)



def _post(path, payload, *, accept="application/json", timeout=120):
    url = f"{BASE}{path}"
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": accept,
            "X-API-Key": TOKEN,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return None, f"transport_error: {type(exc).__name__}"


def _get(path, timeout=30):
    request = urllib.request.Request(f"{BASE}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return None, f"transport_error: {type(exc).__name__}"


RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not condition else ""))
    return bool(condition)


def main():
    print(f"target: {BASE}")
    if not TOKEN:
        print("IMAGINATOR_AUTH_TOKEN is not set; every authenticated probe will fail.")

    status, body = _get("/health")
    health = {}
    try:
        health = json.loads(body)
    except Exception:  # noqa: BLE001
        pass
    check("health endpoint reachable", status == 200 and health.get("status") == "healthy",
          f"status={status}")

    # ── Conservative: the Researcher must be skipped by policy ────────────
    status, raw = _post("/analyze", {
        "resume_text": SOURCE,
        "job_ad": JOB_AD,
        "source_documents": [{"source_id": "document-0", "text": SOURCE,
                              "kind": "candidate_document"}],
        "skill_inference_mode": "conservative",
    })
    conservative = {}
    try:
        conservative = json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    check("conservative request accepted", status in (200, 409),
          f"status={status} body={raw[:160]}")
    check("conservative reports its mode",
          conservative.get("skill_inference_mode") in (None, "conservative"),
          f"mode={conservative.get('skill_inference_mode')}")
    check("conservative carries no bridges",
          not conservative.get("skill_bridges"),
          f"bridges={conservative.get('skill_bridges')!r}")

    # ── Invalid mode must be refused, never silently defaulted ────────────
    status, raw = _post("/analyze", {
        "resume_text": SOURCE,
        "job_ad": JOB_AD,
        "skill_inference_mode": "balanced",
    })
    check("invalid mode is rejected", status in (400, 422),
          f"status={status} body={raw[:160]}")

    # ── Omitted mode takes the documented default ─────────────────────────
    status, raw = _post("/analyze", {
        "resume_text": SOURCE,
        "job_ad": JOB_AD,
        "source_documents": [{"source_id": "document-0", "text": SOURCE,
                              "kind": "candidate_document"}],
    })
    omitted = {}
    try:
        omitted = json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    check("omitted mode is accepted", status in (200, 409),
          f"status={status} body={raw[:160]}")
    check("omitted mode resolves to imaginative",
          omitted.get("skill_inference_mode") in (None, "imaginative"),
          f"mode={omitted.get('skill_inference_mode')}")

    # ── Withheld results still report which policy ran ────────────────────
    # A withheld analysis must not hide the mode: the caller selected it and is
    # entitled to know whether that choice was honoured.
    status, raw = _post("/analyze", {
        "resume_text": SOURCE,
        "job_ad": JOB_AD,
        "source_documents": [{"source_id": "document-0", "text": SOURCE,
                              "kind": "candidate_document"}],
        "skill_inference_mode": "conservative",
    })
    withheld = {}
    try:
        withheld = json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    if withheld.get("correction_status") == "withheld":
        check("a withheld result reports its mode",
              withheld.get("skill_inference_mode") == "conservative",
              f"mode={withheld.get('skill_inference_mode')!r}")
        check("a withheld result reports its policy version",
              withheld.get("skill_inference_policy_version") == "conservative-v1",
              f"policy={withheld.get('skill_inference_policy_version')!r}")
    else:
        print("[SKIP] withhold-mode check — this fixture completed instead of "
              "withholding")

    # ── Export containment: no receipt => refusal ─────────────────────────
    status, raw = _post("/resume/export", {"final_written_section_markdown": DRAFT})
    check("export without a receipt is refused", status == 409,
          f"status={status} body={raw[:160]}")

    # ── Receipt binding, when the run actually produced an approval ───────
    receipt = (omitted.get("publication_receipt") or "")
    if not receipt:
        # The small fixture can never clear the 80-word gate, so try the fuller
        # one before reporting a skip. A skip is still correct when the LIVE
        # REVIEWER withholds: that is a semantic judgement, and pretending it is
        # a failure would push us to weaken a gate to make a probe go green.
        print("[info] small fixture produced no receipt; trying the fuller fixture")
        status, raw = _post("/analyze", {
            "resume_text": FULL_SOURCE,
            "job_ad": JOB_AD,
            "source_documents": [{"source_id": "document-0", "text": FULL_SOURCE,
                                  "kind": "candidate_document"}],
        })
        full = {}
        try:
            full = json.loads(raw)
        except Exception:  # noqa: BLE001
            pass
        check("receipt-capable fixture is accepted", status in (200, 409),
              f"status={status} body={raw[:200]}")
        check("receipt-capable fixture reports a mode",
              full.get("skill_inference_mode") in (None, "imaginative"),
              f"mode={full.get('skill_inference_mode')}")
        receipt = full.get("publication_receipt") or ""
        if receipt:
            omitted = full

    if receipt:
        approved = omitted.get("approved_export_payload") or {}
        payload = dict(approved)
        payload["publication_receipt"] = receipt
        payload["final_written_section_markdown"] = omitted.get(
            "final_written_section_markdown", "")
        state = omitted.get("skill_inference_policy_state") or {
            "skill_inference_mode": "imaginative",
            "skill_inference_policy_version": "imaginative-v1",
            "surviving_bridges": [],
        }
        check("an approved result carries a policy state",
              bool(omitted.get("skill_inference_policy_state")),
              "policy state absent on an approved imaginative result")

        # 1. The untampered approval must be accepted. This is the half that
        #    proves the receipt works at all; refusing everything would also
        #    'pass' a tamper-only test.
        clean = json.loads(json.dumps(payload))
        clean["skill_inference_policy_state"] = state
        status, raw = _post("/resume/export", clean)
        clean_json = {}
        try:
            clean_json = json.loads(raw)
        except Exception:  # noqa: BLE001
            pass
        credentials_ok = status == 200 and bool(clean_json.get("pdf_base64")) \
            and bool(clean_json.get("docx_base64"))
        check("an approved export returns real bytes for both formats",
              credentials_ok,
              f"status={status} keys={sorted(clean_json)}")
        if credentials_ok:
            check("the exported filename is derived from the approval",
                  bool(clean_json.get("filename")),
                  f"filename={clean_json.get('filename')!r}")

        # 2. The same approval with an altered policy state must be refused.
        tampered = json.loads(json.dumps(state))
        tampered["skill_inference_mode"] = "conservative"
        payload["skill_inference_policy_state"] = tampered

        status, raw = _post("/resume/export", payload)
        body_json = {}
        try:
            body_json = json.loads(raw)
        except Exception:  # noqa: BLE001
            pass
        reasons = body_json.get("reasons") or []
        check("tampered policy state is refused",
              status == 409 and any("policy" in str(r) for r in reasons),
              f"status={status} reasons={reasons}")

        # 3. Dropping the state entirely must also be refused (v5 requires it).
        dropped = json.loads(json.dumps(clean))
        dropped.pop("skill_inference_policy_state", None)
        status, raw = _post("/resume/export", dropped)
        dropped_json = {}
        try:
            dropped_json = json.loads(raw)
        except Exception:  # noqa: BLE001
            pass
        dropped_reasons = dropped_json.get("reasons") or []
        check("a v5 receipt presented without its policy state is refused",
              status == 409 and any("policy" in str(r) for r in dropped_reasons),
              f"status={status} reasons={dropped_reasons}")

        # 4. Content tamper must still lose to the existing content binding.
        altered = json.loads(json.dumps(clean))
        altered["final_written_section_markdown"] = (
            altered.get("final_written_section_markdown", "")
            + "\n- Increased regional revenue by 40%.\n"
        )
        status, raw = _post("/resume/export", altered)
        altered_json = {}
        try:
            altered_json = json.loads(raw)
        except Exception:  # noqa: BLE001
            pass
        check("appended content is refused even with a valid receipt",
              status == 409,
              f"status={status} reasons={altered_json.get('reasons')}")
    else:
        # Not a failure: a synthetic fixture may be withheld by the live reviewer.
        # The offline suite proves v5 issuance and tamper refusal; this probe only
        # reports whether a receipted approval was observable on this run.
        print("[SKIP] receipt/policy-tamper check — both synthetic fixtures were "
              "withheld by the live reviewer, so no approval existed to present "
              "(offline suite covers receipt v5 and tamper refusal)")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} probes passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
