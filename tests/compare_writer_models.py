#!/usr/bin/env python3
"""
A/B the Imaginator *writer* stage (Drafter) across OpenRouter models.

The production writer is stages.drafter.Drafter using OR_SLUG_DRAFTER
(pipeline_config.OR_SLUG_DRAFTER, currently anthropic/claude-3.5-haiku).

Usage (from repo root, with keys in env):
  set -a && source /path/to/.env_shared_secrets && set +a
  cd Imagination_Resume_Module && python tests/compare_writer_models.py

Optional:
  python tests/compare_writer_models.py --models qwen/qwen3.6-plus-preview minimax/minimax-m2.1
  python tests/compare_writer_models.py --out /tmp/writer_ab.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Package imports expect Imagination_Resume_Module on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

# Load workspace secrets if present (non-fatal if missing)
load_dotenv(_ROOT.parent / ".env_shared_secrets")
load_dotenv(_ROOT / ".env")

from llm_client_adapter import LLMClientAdapter
from pipeline_config import TEMPERATURES
from stages.drafter import Drafter

logging.basicConfig(level=logging.WARNING)

# Slugs: verify via https://openrouter.ai/api/v1/models
# Production drafter vs Grok fast (cost experiment); plus legacy comparators.
DEFAULT_MODELS = [
    "anthropic/claude-3.5-haiku",
    "x-ai/grok-4.1-fast",
    "qwen/qwen3.6-plus:free",
    "minimax/minimax-m2.1",
]


def _sample_payload() -> Tuple[List[Dict], str, Dict[str, Any]]:
    """Fixed resume slice + job ad + researcher output for repeatable comparison."""
    experiences = [
        {
            "company": "Nimbus Analytics",
            "role": "Software Engineer",
            "duration": "2021 – 2024",
            "location": "Remote",
            "bullets": [
                "Built internal REST APIs for customer reporting",
                "Helped migrate three services from EC2 to EKS",
                "Optimized PostgreSQL queries for executive dashboards",
            ],
        },
        {
            "company": "Riverloop Labs",
            "role": "Junior Developer",
            "duration": "2019 – 2021",
            "location": "Austin, TX",
            "bullets": [
                "Wrote Python scripts for data cleanup",
                "Fixed bugs in the billing microservice",
            ],
        },
    ]
    job_ad = """
    Armada AI — Senior Backend Engineer (remote)
    We need someone to own our inference gateway: sub-50ms p99, 99.95% SLA,
    Kubernetes, gRPC, PostgreSQL, Redis. Lead incident response and cost optimization.
    Must have shipped high-traffic APIs and mentored engineers.
    """.strip()

    research_data = {
        "implied_metrics": ["p99 latency", "API throughput", "SLA uptime", "cost per token"],
        "domain_vocab": ["Kubernetes", "gRPC", "Redis", "PostgreSQL", "incident response"],
        "implied_skills": ["Python", "Go", "observability", "SRE practices"],
        "work_archetypes": ["High-scale API platform"],
    }
    return experiences, job_ad, research_data


def _company_integrity_score(
    rewritten: List[Dict], allowed_companies: List[str]
) -> float:
    """1.0 if every company string is a substring match of an allowed company (case-relaxed)."""
    if not rewritten:
        return 0.0
    allowed_l = [c.lower() for c in allowed_companies]
    ok = 0
    for exp in rewritten:
        c = (exp.get("company") or "").strip().lower()
        if any(c == a or c in a or a in c for a in allowed_l):
            ok += 1
    return ok / len(rewritten)


def _forbidden_job_ad_leak(job_ad: str, bullets: List[str]) -> int:
    """Count bullets containing distinctive job-ad-only tokens (Armada, inference gateway)."""
    leaks = 0
    markers = ["armada", "inference gateway", "99.95%", "99.95"]
    ja = job_ad.lower()
    for b in bullets:
        bl = b.lower()
        for m in markers:
            if m in ja and m in bl:
                leaks += 1
                break
    return leaks


def _composite_metrics(
    draft: Dict[str, Any], job_ad: str, allowed_companies: List[str]
) -> Dict[str, Any]:
    rewritten = draft.get("rewritten_experiences") or []
    all_bullets: List[str] = []
    for exp in rewritten:
        for b in exp.get("bullets") or []:
            if isinstance(b, str):
                all_bullets.append(b)

    return {
        "total_bullets": draft.get("total_bullets", len(all_bullets)),
        "quantification_score": draft.get("quantification_score", 0.0),
        "quantified_bullets": draft.get("quantified_bullets", 0),
        "has_placeholders": bool(draft.get("has_placeholders")),
        "fallback": bool(draft.get("fallback")),
        "company_integrity": round(
            _company_integrity_score(rewritten, allowed_companies), 3
        ),
        "job_ad_leak_count": _forbidden_job_ad_leak(job_ad, all_bullets),
        "avg_bullet_len": (
            round(sum(len(b) for b in all_bullets) / len(all_bullets), 1)
            if all_bullets
            else 0.0
        ),
    }


def _score_for_ranking(m: Dict[str, Any]) -> float:
    """
    Higher is better. Heuristic only — read outputs manually for tone/STAR quality.
    Penalize placeholders, fallbacks, and copying the target employer into bullets.
    """
    if m.get("error"):
        return -1e6
    q = float(m.get("quantification_score", 0.0))
    integrity = float(m.get("company_integrity", 0.0))
    leaks = int(m.get("job_ad_leak_count", 0))
    ph = 1.0 if m.get("has_placeholders") else 0.0
    fb = 1.0 if m.get("fallback") else 0.0
    bullets = int(m.get("total_bullets", 0))
    # Weights: quantification and truthfulness dominate
    return (
        4.0 * q
        + 2.0 * integrity
        + 0.02 * min(bullets, 40)
        - 3.0 * leaks
        - 2.0 * ph
        - 1.5 * fb
    )


async def _run_one_model(
    drafter: Drafter,
    model: str,
    experiences: List[Dict],
    job_ad: str,
    research_data: Dict[str, Any],
    http_timeout: int,
) -> Dict[str, Any]:
    drafter.model = model
    drafter.fallback_models = []  # no OpenRouter chain — exact model only
    drafter.timeout = http_timeout

    allowed = [e.get("company", "") for e in experiences if e.get("company")]
    t0 = time.perf_counter()
    try:
        draft = await drafter.draft(
            experiences=experiences,
            job_ad=job_ad,
            research_data=research_data,
            golden_bullets=None,
            tone_instruction="Maintain a standard, professional corporate tone.",
            temperature_override=TEMPERATURES["drafter"],
            extracted_job_title="Senior Backend Engineer",
        )
        elapsed = round(time.perf_counter() - t0, 2)
        metrics = _composite_metrics(draft, job_ad, allowed)
        metrics["model"] = model
        metrics["seconds"] = elapsed
        metrics["model_requested"] = draft.get("model_requested", model)
        metrics["sample_bullets"] = [
            b
            for exp in (draft.get("rewritten_experiences") or [])[:2]
            for b in (exp.get("bullets") or [])[:2]
            if isinstance(b, str)
        ]
        metrics["rank_score"] = round(_score_for_ranking(metrics), 4)
        return metrics
    except Exception as e:
        return {
            "model": model,
            "error": str(e),
            "seconds": round(time.perf_counter() - t0, 2),
            "rank_score": -1e6,
        }


async def main_async(args: argparse.Namespace) -> List[Dict[str, Any]]:
    key = os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        print(
            "Missing OPENROUTER_API_KEY_1 (or OPENROUTER_API_KEY). "
            "Source your env file first.",
            file=sys.stderr,
        )
        sys.exit(2)

    experiences, job_ad, research_data = _sample_payload()
    client = LLMClientAdapter(api_key=key)
    drafter = Drafter(client)

    results: List[Dict[str, Any]] = []
    for model in args.models:
        print(f"\n=== Drafter model: {model} ===", flush=True)
        row = await _run_one_model(
            drafter,
            model,
            experiences,
            job_ad,
            research_data,
            args.timeout,
        )
        results.append(row)

    ranked = sorted(
        [r for r in results if "rank_score" in r],
        key=lambda x: x["rank_score"],
        reverse=True,
    )
    print("\n========== SUMMARY (heuristic rank_score; verify prose manually) ==========")
    for r in ranked:
        if r.get("error"):
            print(f"  FAIL {r['model']}: {r['error']}")
            continue
        print(
            f"  {r['model']}: rank_score={r['rank_score']} "
            f"quant={r['quantification_score']} "
            f"bullets={r['total_bullets']} "
            f"integrity={r['company_integrity']} "
            f"leaks={r['job_ad_leak_count']} "
            f"placeholders={r['has_placeholders']} "
            f"{r['seconds']}s"
        )
    if ranked and not any(r.get("error") for r in ranked):
        winner = ranked[0]["model"]
        print(f"\nHeuristic winner: {winner}")
        print("(Read sample_bullets in JSON output for subjective quality.)")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Imaginator Drafter (writer) models")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="OpenRouter model slugs to compare",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Write full JSON results to this path",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request HTTP timeout seconds",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
