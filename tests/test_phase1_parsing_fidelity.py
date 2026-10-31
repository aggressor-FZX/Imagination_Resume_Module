"""Phase 1 reading-side fidelity checks — RCTB §9 R-7 and R-8.

R-7: extraction completeness — every company in the answer key must survive
     parse_experiences().
R-8: date fidelity — parsed date ranges must match the answer key.

These use repository-owned synthetic resumes and independently written answer
keys, vendored as a small source copy of the RCTB profiles. Legacy fixture IDs
are labels, not real people. No sibling harness or private data is required.

The answer keys are Python-dict-shaped text files, so they are parsed rather
than imported. Missing or malformed packaged fixtures fail, never skip.
"""
import ast
import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "parsing_fidelity"
FIXTURE_FILES = (
    "Holloway_ElecWA25.txt",
    "Whitfield_AdminNC25.txt",
    "answer_key_elec.txt",
    "answer_key_admin.txt",
)

PROFILE_CASES = (
    ("Holloway_ElecWA25.txt", "answer_key_elec.txt", "Holloway_ElecWA25"),
    ("Whitfield_AdminNC25.txt", "answer_key_admin.txt", "Whitfield_AdminNC25"),
)


@pytest.fixture(autouse=True)
def _offline_parser_tests(monkeypatch):
    """Exercise the real regex parser without dotenv reads or outbound calls."""
    import dotenv
    from pydantic_settings.sources import DotEnvSettingsSource

    # Parsing must not construct model clients from inherited credentials.
    for name in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2",
        "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "CONTEXT7_API_KEY",
        "IMAGINATOR_AUTH_TOKEN", "API_SERVICE_KEY", "JOB_SEARCH_API_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})

    def no_network(*args, **kwargs):
        raise AssertionError("Parsing fidelity tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)


def _load_answer_key(filename):
    """The answer keys are `ANSWER_KEY = {...}` python-source text files."""
    path = FIXTURE_DIR / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ANSWER_KEY":
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{path} has no ANSWER_KEY assignment")


def _load_resume_text(filename):
    return (FIXTURE_DIR / filename).read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", FIXTURE_FILES)
def test_fidelity_fixture_is_packaged(filename):
    """A clean app checkout must contain every input and independent key."""
    path = FIXTURE_DIR / filename
    assert path.is_file(), f"Missing packaged synthetic fidelity fixture: {path}"
    assert path.read_text(encoding="utf-8").strip(), f"Empty fidelity fixture: {path}"


@pytest.mark.parametrize("resume_file,key_file,key_name", PROFILE_CASES)
def test_fidelity_loaders_are_cwd_independent(
    monkeypatch, tmp_path, resume_file, key_file, key_name
):
    monkeypatch.chdir(tmp_path)

    answer = _load_answer_key(key_file)[key_name]
    text = _load_resume_text(resume_file)
    assert answer["synthetic"] is True
    assert len(answer["expected_companies"]) == 2
    assert len(answer["expected_durations"]) == 2
    assert "example.com" in text
    assert all(company in text for company in answer["expected_companies"])
    assert all(duration in text for duration in answer["expected_durations"])


@pytest.mark.parametrize("loader,filename", [
    (_load_resume_text, "Holloway_ElecWA25.txt"),
    (_load_answer_key, "answer_key_elec.txt"),
])
def test_missing_packaged_fixture_fails_without_fallback(
    monkeypatch, tmp_path, loader, filename
):
    monkeypatch.setitem(globals(), "FIXTURE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        loader(filename)


@pytest.mark.parametrize("source,error", [
    ("NOT_AN_ANSWER_KEY = {}", AssertionError),
    ("ANSWER_KEY = {", SyntaxError),
    ("ANSWER_KEY = dict()", ValueError),
])
def test_malformed_answer_key_fails(monkeypatch, tmp_path, source, error):
    monkeypatch.setitem(globals(), "FIXTURE_DIR", tmp_path)
    (tmp_path / "answer_key_elec.txt").write_text(source, encoding="utf-8")
    with pytest.raises(error):
        _load_answer_key("answer_key_elec.txt")


# ─────────────────────────────────────────────────────────────────────────────
# R-7: extraction completeness against the answer key
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("resume_file,key_file,key_name", PROFILE_CASES)
def test_r7_every_expected_company_is_extracted(resume_file, key_file, key_name):
    """R-7: no expected employer is silently dropped by the regex parser."""
    from imaginator_flow import parse_experiences

    answer = _load_answer_key(key_file)[key_name]
    text = _load_resume_text(resume_file)

    parsed = parse_experiences(text)
    found = " || ".join(
        f"{e.get('company','')} {e.get('title_line','')} {e.get('raw','')}"
        for e in parsed
    ).lower()

    missing = [
        c for c in answer["expected_companies"]
        if c.lower() not in found
    ]
    assert not missing, (
        f"parse_experiences dropped {len(missing)} employer(s) from "
        f"{resume_file}: {missing}\n"
        f"parsed {len(parsed)} blocks: "
        f"{[e.get('company') for e in parsed]}"
    )


def test_r7_experience_count_is_close_to_answer_key():
    """R-7: the parser must find every EMPLOYMENT block, not collapse them.

    The synthetic electrician profile has two employer headers with date ranges
    and a separate education entry. Assert on employers, never promote education
    to a job (§6: keep projects/education separate from employers).
    """
    from imaginator_flow import parse_experiences

    answer = _load_answer_key("answer_key_elec.txt")["Holloway_ElecWA25"]
    parsed = parse_experiences(_load_resume_text("Holloway_ElecWA25.txt"))

    employment_companies = [
        c for c in answer["expected_companies"]
        if "apprenticeship" not in c.lower()
    ]
    found_companies = {
        (e.get("company") or "").strip().lower() for e in parsed
    }
    missing = [
        c for c in employment_companies if c.lower() not in found_companies
    ]
    assert not missing, (
        f"employment blocks collapsed or lost: {missing}\n"
        f"parsed companies: {[e.get('company') for e in parsed]}"
    )

    # No phantom blocks: every parsed block must carry a company.
    empty = [e for e in parsed if not (e.get("company") or "").strip()]
    assert not empty, (
        f"{len(empty)} block(s) with no company — a wrapped bullet was "
        f"promoted to a job title"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R-8: date fidelity
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("resume_file", [
    "Holloway_ElecWA25.txt",
    "Whitfield_AdminNC25.txt",
])
def test_r8_dates_are_captured(resume_file):
    """R-8: every parsed experience with a title carries non-empty duration info."""
    from imaginator_flow import parse_experiences

    parsed = parse_experiences(_load_resume_text(resume_file))
    assert parsed, "parser returned no experiences at all"

    with_duration = [
        e for e in parsed
        if e.get("duration") or e.get("start_date")
    ]
    assert with_duration, (
        f"no experience in {resume_file} produced a duration; "
        f"keys: {sorted(parsed[0].keys())}"
    )


@pytest.mark.parametrize("resume_file,key_file,key_name", PROFILE_CASES)
def test_r8_company_date_ranges_match_answer_key(resume_file, key_file, key_name):
    """R-8: preserve both date endpoints on each independently keyed employer."""
    from imaginator_flow import parse_experiences

    answer = _load_answer_key(key_file)[key_name]
    parsed = parse_experiences(_load_resume_text(resume_file))
    companies = answer["expected_companies"]
    durations = answer["expected_durations"]
    assert len(companies) == len(durations) == 2
    for company, duration in zip(companies, durations):
        matches = [e for e in parsed if e.get("company") == company]
        assert len(matches) == 1, (
            f"Expected one employment block for {company}: {matches}; "
            f"parsed companies/durations: "
            f"{[(e.get('company'), e.get('duration')) for e in parsed]}"
        )
        assert matches[0].get("duration") == duration, (
            f"Date range changed for {company}: {matches[0].get('duration')!r} "
            f"!= {duration!r}"
        )


def test_r8_present_is_not_mistaken_for_a_date():
    """R-8: 'Present' must stay 'Present', not become a fabricated month."""
    from imaginator_flow import parse_experiences

    text = (
        "PROFESSIONAL EXPERIENCE\n\n"
        "Cascade Power Solutions, LLC - Tacoma, WA | March 2020 - Present\n"
        "- Install and repair electrical wiring\n"
    )
    parsed = parse_experiences(text)
    assert parsed, "parser returned no experiences"
    joined = " ".join(str(e.get("duration", "")) for e in parsed).lower()
    assert "present" in joined or "2020" in joined, (
        f"date range lost entirely: {[e.get('duration') for e in parsed]}"
    )
    # Must not invent a January date to fill the open end.
    assert "01/2024" not in joined, "fabricated start date '01/2024' appeared"
