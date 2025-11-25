import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "imaginator_flow.py"
SAMPLE_RESUME = ROOT / "sample_resume.txt"
SAMPLE_JOB_AD = ROOT / "sample_job_ad.txt"
SAMPLE_SKILLS = ROOT / "sample_skills.json"
SAMPLE_INSIGHTS = ROOT / "sample_insights.json"


def _extract_json_from_stdout(text: str):
    # Try all positions of '{' to find a valid full JSON that consumes till the end
    positions = [i for i, ch in enumerate(text) if ch == "{"]
    last_err = None
    for pos in positions:
        snippet = text[pos:].strip()
        try:
            obj = json.loads(snippet)
            return obj
        except json.JSONDecodeError as e:
            last_err = e
            continue
    raise AssertionError(f"No valid JSON object could be parsed from output. Last error: {last_err}\nOutput was:\n{text}")


def test_e2e_graceful_degradation_schema_validates(tmp_path):
    """
    Run the script end-to-end with invalid API keys to trigger graceful degradation
    and ensure the final output is valid JSON matching the schema.
    """
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "invalid"
    env["ANTHROPIC_API_KEY"] = "invalid"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--resume",
        str(SAMPLE_RESUME),
        "--target_job_ad",
        SAMPLE_JOB_AD.read_text(),
        "--extracted_skills_json",
        str(SAMPLE_SKILLS),
        "--domain_insights_json",
        str(SAMPLE_INSIGHTS),
        "--confidence_threshold",
        "0.6",
    ]

    # Use a timeout safeguard (pytest should kill if it hangs, but be safe)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    # The script prints progress logs and then final JSON to stdout
    data = _extract_json_from_stdout(proc.stdout)

    # Basic structure checks
    assert "experiences" in data
    assert "aggregate_skills" in data
    assert "processed_skills" in data
    assert "domain_insights" in data
    assert "gap_analysis" in data
    assert "suggested_experiences" in data

    # suggested_experiences shape
    se = data["suggested_experiences"]
    assert isinstance(se.get("bridging_gaps"), list)
    assert isinstance(se.get("metric_improvements"), list)

    # gap_analysis is a stringified JSON
    assert isinstance(data["gap_analysis"], str)
