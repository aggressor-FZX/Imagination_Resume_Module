#!/usr/bin/env python3
"""
Multi-file resume simulation + Drafter (writer) model comparison.

1. Loads combined resume text (multiple PDFs concatenated with ===== markers).
2. Parses experiences like production (imaginator_flow.parse_experiences).
3. Builds synthetic upstream payloads (Hermes-style entities, FastSVM-style predictions)
   for documentation — the Drafter still uses the same prompts as production.
4. Runs Researcher once (shared) then Drafter per model with fallback_models=[].

Example:
  set -a && source ../.env_shared_secrets && set +a
  python tests/compare_writer_multiresume.py \\
    --resume /mnt/c/Users/jeffd/latexRoot/My_resume_tests/combined_resumes.txt \\
    --out-dir ./reports

  # Head-to-head: production Haiku vs Grok 4.1 Fast (same prompt corpus)
  python tests/compare_writer_multiresume.py \\
    --models anthropic/claude-3.5-haiku x-ai/grok-4.1-fast \\
    --out-dir ./reports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT.parent / ".env_shared_secrets")
load_dotenv(_ROOT / ".env")

from imaginator_flow import extract_skills_from_experience, parse_experiences
from llm_client_adapter import LLMClientAdapter
from role_title_sanitizer import is_plausible_employer_name, is_valid_extracted_job_title
from pipeline_config import TEMPERATURES
from stages.drafter import Drafter
from stages.researcher import Researcher

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class _DrafterTokenBoostAdapter:
    """Reasoning models can burn the default max_tokens before emitting JSON; raise ceiling."""

    def __init__(self, inner: LLMClientAdapter, max_tokens: int = 32768):
        self._inner = inner
        self._max_tokens = max_tokens

    async def call_llm_async(self, *args, **kwargs):
        kwargs.setdefault("max_tokens", self._max_tokens)
        return await self._inner.call_llm_async(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

DEFAULT_RESUME = (
    "/mnt/c/Users/jeffd/latexRoot/My_resume_tests/combined_resumes.txt"
)
# Production writer = pipeline_config.OR_SLUG_DRAFTER (anthropic/claude-3.5-haiku).
# Other slugs: experimental / user-requested (OpenRouter availability varies).
DEFAULT_MODELS = [
    "anthropic/claude-3.5-haiku",
    "x-ai/grok-4.1-fast",
    "xiaomi/mimo-v2-flash",
    "qwen/qwen-3.6-plus-preview",
    "qwen/qwen3.6-plus:free",
    "minimax/minimax-m2.1",
]

# OpenRouter catalog IDs (2026-04) for a user-supplied benchmark matrix — several “marketing”
# slugs differ from live `GET /api/v1/models` ids (hyphen vs dot, provider prefix, etc.).
BENCHMARK_MODELS_USER_MATRIX = [
    "qwen/qwen3.5-9b",
    "inception/mercury-2",
    "minimax/minimax-m2.7",  # not minimax-m2-7
    "openai/gpt-5.4-mini",  # not gpt-5.4-mini-medium
    "z-ai/glm-5",
    "anthropic/claude-sonnet-4.6",  # not …-non-reasoning-low-effort (not listed)
    "openai/gpt-5.4-nano",  # not …-nano-medium
    "openai/gpt-5.4",  # not …-non-reasoning
    "z-ai/glm-5-turbo",  # closest listed to “glm-5 non-reasoning”
    "google/gemini-3-flash-preview",  # not …-reasoning
    "moonshotai/kimi-k2.5",  # not kimi/kimi-k2.5-non-reasoning
]

# Fictitious employer wording to detect job-ad plagiarism in bullets
JOB_AD_MARKERS = (
    "skyharbor",
    "nebulaforge",
    "quantmesh",
    "sub-40ms p99",
)


def _preprocess_combined_text(text: str) -> str:
    """Normalize multi-file headers (===== file.pdf =====) to spacing only."""
    lines_out = []
    for line in text.splitlines():
        if re.match(r"^=+\s*.+\.(pdf|docx)\s*=+$", line.strip(), re.I):
            lines_out.append("\n\n--- combined upload ---\n\n")
            continue
        lines_out.append(line)
    return "\n".join(lines_out)


def _extract_bullets_from_raw(raw: str) -> List[str]:
    bullets = []
    for line in (raw or "").split("\n"):
        line = line.strip()
        if re.match(r"^[\s]*[•\-\*\–\—\►\◦\▪\●]", line) or re.match(
            r"^[\s]*\d+[\.\)]\s", line
        ):
            cleaned = re.sub(
                r"^[\s]*(?:[•\-\*\–\—\►\◦\▪\●]|\d+[\.\)])\s*", "", line
            ).strip()
            if cleaned:
                bullets.append(cleaned)
    return bullets


def _split_role_company(title_line: str) -> Tuple[str, str]:
    """Best-effort role / company from a single title line."""
    tl = (title_line or "").strip()
    if not tl:
        return "Unknown", "Unknown"
    # "Role, Org (dates)" or "Role, Org"
    m = re.match(r"^(.+?),\s*(.+)$", tl)
    if m:
        role = m.group(1).strip()
        rest = m.group(2).strip()
        rest = re.sub(r"(?i)\binternship\b", "", rest).strip()
        rest = re.sub(
            r"\s*\([^)]*(?:present|internship|\d{4})[^)]*\)\s*$",
            "",
            rest,
            flags=re.I,
        ).strip()
        rest = re.sub(
            r"\s*\d{1,2}/\d{2,4}\s*[-–]\s*(?:\d{1,2}/\d{2,4}|present).*$",
            "",
            rest,
            flags=re.I,
        ).strip()
        return role[:120], (rest or "Unknown")[:180]
    return tl[:120], "Unknown"


def _unique_plausible_orgs(
    experiences: List[Dict[str, Any]], *, limit: int = 20
) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for e in experiences:
        c = (e.get("company") or "").strip()
        if (
            not c
            or c == "Unknown"
            or not is_plausible_employer_name(c)
            or c in seen
        ):
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _dedupe_experiences(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        co = (r.get("company") or "").strip()
        if not is_plausible_employer_name(co):
            r = {**r, "company": "Unknown"}
        key = (
            (r.get("role") or "").lower()[:80],
            (r.get("company") or "").lower()[:100],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def experiences_for_drafter(
    parsed: List[Dict[str, Any]],
    max_roles: int = 10,
    max_bullets_per_role: int = 6,
    max_bullet_chars: int = 380,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp in parsed:
        raw = exp.get("raw") or ""
        role, company = _split_role_company(exp.get("title_line") or "")
        bullets = _extract_bullets_from_raw(raw)
        if not bullets and exp.get("body"):
            bullets = [exp["body"][:max_bullet_chars]]
        bullets = [b[:max_bullet_chars] for b in bullets[:max_bullets_per_role]]
        if not is_valid_extracted_job_title(role) and role.strip():
            bullets = [role.strip()] + bullets
            role = "Professional"
        rows.append(
            {
                "company": company,
                "role": role,
                "duration": exp.get("duration") or "",
                "location": exp.get("location") or "",
                "bullets": bullets,
            }
        )
    rows = _dedupe_experiences(rows)
    return rows[:max_roles]


def build_upstream_simulation(
    experiences: List[Dict[str, Any]], resume_text: str
) -> Dict[str, Any]:
    """Synthetic Hermes + FastSVM-style payloads (for your review; not sent to OpenRouter)."""
    all_skills: Set[str] = set()
    for exp in experiences:
        blob = " ".join(
            [exp.get("role", ""), exp.get("company", "")]
            + list(exp.get("bullets") or [])
        )
        all_skills |= extract_skills_from_experience(blob)
    for chunk in resume_text.split("--- combined upload ---"):
        all_skills |= extract_skills_from_experience(chunk[:8000])

    skill_list = sorted(all_skills)
    fastsvm_skills = [
        {"skill": s.lower().replace(" ", "_"), "confidence": round(0.75 + 0.02 * (i % 5), 2)}
        for i, s in enumerate(skill_list[:25])
    ]
    return {
        "hermes_style": {
            "note": "Simulated structured extraction from document text",
            "aggregate_entities": {
                "organizations": _unique_plausible_orgs(experiences, limit=20),
                "roles": [
                    role
                    for role in (e.get("role") for e in experiences[:15])
                    if role
                    and (
                        is_valid_extracted_job_title(role)
                        or role.strip() == "Professional"
                    )
                ],
                "skills_flat": skill_list[:60],
            },
        },
        "fastsvm_style": {
            "title": "Analytics / Data Engineering",
            "canonical_title": "Senior Analytics Engineer",
            "overall_confidence": 0.86,
            "skills": fastsvm_skills,
        },
    }


def sample_job_ad() -> str:
    return """
SkyHarbor Analytics — Senior Analytics Engineer (Remote, US)

We ingest 4B+ events/day into NebulaForge, our real-time lakehouse. You will own
pipelines (Python, SQL, Spark), QuantMesh metric stores, and sub-40ms p99 dashboards
for executive KPIs. Partner with ML on feature stores; enforce data contracts;
mentor two junior engineers. Requires AWS or GCP, Airflow or Dagster, and a track
record of measurable reliability wins — not slide decks.

Nice-to-have: experience with government or regulated telemetry, on-call rotation,
and cost guardrails for big batch + streaming stacks.
""".strip()


def style_rubric(draft: Dict[str, Any], job_ad: str, allowed_companies: List[str]) -> Dict[str, Any]:
    rewritten = draft.get("rewritten_experiences") or []
    all_bullets: List[str] = []
    for exp in rewritten:
        for b in exp.get("bullets") or []:
            if isinstance(b, str):
                all_bullets.append(b)

    weak = re.compile(
        r"\b(strategized|participated|pioneered|synergized|utilized extensively)\b",
        re.I,
    )
    weak_hits = sum(1 for b in all_bullets if weak.search(b))

    md_artifacts = sum(
        1 for b in all_bullets if re.search(r"[*_]{2,}|^#+\s", b)
    )

    leaks = 0
    ja = job_ad.lower()
    for b in all_bullets:
        bl = b.lower()
        for m in JOB_AD_MARKERS:
            if m in ja and m in bl:
                leaks += 1
                break

    allowed_l = [c.lower() for c in allowed_companies if c and c != "Unknown"]
    foreign_company_hits = 0
    for exp in rewritten:
        c = (exp.get("company") or "").strip().lower()
        if not c:
            continue
        if not any(c == a or c in a or a in c for a in allowed_l):
            foreign_company_hits += 1

    return {
        "bullet_count": len(all_bullets),
        "quantification_score": draft.get("quantification_score", 0.0),
        "has_placeholders": bool(draft.get("has_placeholders")),
        "fallback": bool(draft.get("fallback")),
        "weak_verb_hits": weak_hits,
        "markdown_noise_lines": md_artifacts,
        "distinctive_job_ad_leaks": leaks,
        "company_field_mismatches_vs_allowlist": foreign_company_hits,
        "avg_bullet_len": (
            round(sum(len(b) for b in all_bullets) / len(all_bullets), 1)
            if all_bullets
            else 0.0
        ),
    }


async def _run_researcher(
    client: LLMClientAdapter, job_ad: str, experiences: List[Dict[str, Any]]
) -> Dict[str, Any]:
    r = Researcher(client)
    return await r.analyze(job_ad, experiences, temperature_override=TEMPERATURES["researcher"])


async def _run_drafter(
    client: LLMClientAdapter,
    model: str,
    experiences: List[Dict[str, Any]],
    job_ad: str,
    research: Dict[str, Any],
    http_timeout: int,
    extracted_title: str,
) -> Dict[str, Any]:
    d = Drafter(client)
    d.model = model
    d.fallback_models = []
    d.timeout = http_timeout
    t0 = time.perf_counter()
    try:
        out = await d.draft(
            experiences=experiences,
            job_ad=job_ad,
            research_data=research,
            golden_bullets=None,
            tone_instruction="Maintain a standard, professional corporate tone.",
            temperature_override=TEMPERATURES["drafter"],
            extracted_job_title=extracted_title,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        out["_wall_seconds"] = elapsed
        out["_model"] = model
        return out
    except Exception as e:
        return {
            "_model": model,
            "_error": str(e),
            "_wall_seconds": round(time.perf_counter() - t0, 2),
            "rewritten_experiences": [],
            "fallback": True,
        }


def _write_markdown(
    path: Path,
    meta: Dict[str, Any],
    job_ad: str,
    research: Dict[str, Any],
    upstream: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Writer (Drafter) comparison — multi-resume simulation",
        "",
        f"Generated: {meta.get('generated_at', '')}",
        f"Source file: `{meta.get('resume_path', '')}`",
        f"Parsed experience blocks (pre-dedupe): {meta.get('parsed_count', 0)}",
        f"Drafter input roles (deduped, capped): {meta.get('drafter_input_roles', 0)}",
        f"Drafter `max_tokens`: {meta.get('drafter_max_tokens', '')}",
        "",
        "## How to read this",
        "",
        "- **Style / formatting**: bullet length, parallelism, weak verbs, stray markdown.",
        "- **Rule following**: `distinctive_job_ad_leaks` (SkyHarbor / NebulaForge / QuantMesh / p99 phrasing),",
        "  `company_field_mismatches_vs_allowlist` (rewritten `company` not matching parsed employers).",
        "- **Quantification**: fraction of bullets with numbers (heuristic; does not prove factual accuracy).",
        "",
        "## Simulated upstream services (review artifacts)",
        "",
        "Hermes-style aggregate entities and FastSVM-style predictions are **synthetic**",
        "reconstructions for documentation; the live Drafter prompt matches production.",
        "",
        "```json",
        json.dumps(upstream, indent=2)[:12000],
        "```",
        "",
        "## Target job ad (fictitious company — leak test)",
        "",
        job_ad,
        "",
        "## Researcher output (shared across models)",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in research.items() if k != "model_used"},
            indent=2,
        )[:8000],
        "```",
        "",
    ]
    for block in results:
        model = block.get("model", "?")
        lines.append(f"## Model: `{model}`")
        rub = block.get("rubric") or {}
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for k, v in rub.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append(f"Wall time: {block.get('seconds', '?')}s")
        if block.get("error"):
            lines.append(f"**Error:** {block['error']}")
            lines.append("")
            continue
        lines.append("")
        lines.append("### Rewritten experiences (markdown preview)")
        lines.append("")
        for exp in (block.get("draft") or {}).get("rewritten_experiences") or []:
            role = exp.get("role", "")
            co = exp.get("company", "")
            lines.append(f"**{role}** — *{co}*")
            for b in exp.get("bullets") or []:
                lines.append(f"- {b}")
            lines.append("")
        lines.append("---")
        lines.append("")
    lines.append("## Run notes (automated)")
    lines.append("")
    for r in results:
        m = r.get("model", "")
        rub = r.get("rubric") or {}
        if m.startswith("minimax") and rub.get("fallback"):
            lines.append(
                f"- **`{m}`**: Drafter fell back (often empty JSON when the provider returns "
                f"reasoning-only or truncated `message.content` at large prompt sizes). "
                f"Try `--drafter-max-tokens 32000` or a different MiniMax provider route."
            )
    lines.append(
        "- Compare **Xiaomi** vs **Qwen** (and others) by reading the bullet sections above; "
        "rubric scores are heuristics only."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    key = os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("Missing OPENROUTER_API_KEY_1", file=sys.stderr)
        sys.exit(2)

    resume_path = Path(args.resume).expanduser()
    if not resume_path.is_file():
        print(f"Resume file not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    raw_text = resume_path.read_text(encoding="utf-8", errors="replace")
    combined = _preprocess_combined_text(raw_text)
    parsed = parse_experiences(combined)
    experiences = experiences_for_drafter(
        parsed,
        max_roles=args.max_roles,
        max_bullets_per_role=args.max_bullets,
    )
    job_ad = sample_job_ad()
    upstream = build_upstream_simulation(experiences, combined)
    extracted_title = "Senior Analytics Engineer"

    allowed_companies = [
        e.get("company", "") for e in experiences if e.get("company")
    ]

    client = LLMClientAdapter(api_key=key)
    research = await _run_researcher(client, job_ad, experiences)

    results: List[Dict[str, Any]] = []
    drafter_client: Any = _DrafterTokenBoostAdapter(
        client, max_tokens=args.drafter_max_tokens
    )
    for model in args.models:
        print(f"\n>>> Drafter: {model}", flush=True)
        draft = await _run_drafter(
            drafter_client,
            model,
            experiences,
            job_ad,
            research,
            args.timeout,
            extracted_title,
        )
        err = draft.pop("_error", None)
        seconds = draft.pop("_wall_seconds", None)
        model_tag = draft.pop("_model", model)
        rubric = style_rubric(draft, job_ad, allowed_companies)
        results.append(
            {
                "model": model_tag,
                "seconds": seconds,
                "error": err,
                "rubric": rubric,
                "draft": draft,
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"writer_multicompare_{ts}"

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resume_path": str(resume_path),
        "parsed_count": len(parsed),
        "drafter_input_roles": len(experiences),
        "models": args.models,
        "job_title_context": extracted_title,
        "drafter_max_tokens": args.drafter_max_tokens,
    }

    json_payload = {
        "meta": meta,
        "job_ad": job_ad,
        "upstream_simulation": upstream,
        "researcher_output": research,
        "drafter_input_experiences": experiences,
        "results": results,
    }
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_path = base.with_suffix(".md")
    _write_markdown(md_path, meta, job_ad, research, upstream, results)

    print(f"\nWrote:\n  {json_path}\n  {md_path}", flush=True)

    # Console verdict: rank by rubric
    def score(r: Dict[str, Any]) -> float:
        if r.get("error"):
            return -1e9
        u = r["rubric"]
        return (
            3.0 * float(u.get("quantification_score", 0))
            - 2.0 * int(u.get("distinctive_job_ad_leaks", 0))
            - 1.5 * int(u.get("weak_verb_hits", 0))
            - 2.0 * int(u.get("company_field_mismatches_vs_allowlist", 0))
            - 0.5 * int(u.get("markdown_noise_lines", 0))
            - (3.0 if u.get("has_placeholders") else 0)
            - (2.0 if u.get("fallback") else 0)
        )

    ranked = sorted(results, key=score, reverse=True)
    print("\n=== Heuristic ranking (higher is better) ===")
    for r in ranked:
        print(f"  {r['model']}: score={score(r):.2f}  {r.get('rubric')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default=DEFAULT_RESUME, help="Path to combined resume text")
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "reports"),
        help="Directory for JSON + Markdown output",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="OpenRouter writer models to compare",
    )
    p.add_argument(
        "--benchmark-user-matrix",
        action="store_true",
        help="Run the 11-model matrix from BENCHMARK_MODELS_USER_MATRIX (catalog-verified slugs).",
    )
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument(
        "--drafter-max-tokens",
        type=int,
        default=32768,
        help="Completion budget for Drafter (reasoning models need headroom)",
    )
    p.add_argument("--max-roles", type=int, default=10)
    p.add_argument("--max-bullets", type=int, default=6)
    ns = p.parse_args()
    if ns.benchmark_user_matrix:
        ns.models = list(BENCHMARK_MODELS_USER_MATRIX)
    return ns


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
