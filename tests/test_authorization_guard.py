"""Once-only authorization, and the render/execute boundary.

Both exist because a screen was run twice. The per-run request ceiling held in each
invocation (20 and 21 against a stated 40) while the aggregate was 41, and nothing in
the code could have stopped a second invocation because `--authorized` was a
stateless boolean.

These tests lock the two structural properties that would have prevented it:

1. **A token can be consumed once.** A second attempt is refused BEFORE any provider
   client exists. The claim is written before the run starts, so a crash still blocks
   a silent retry.
2. **Rendering cannot dispatch.** `render_from_artifact` takes no `ask` callable and
   imports no provider module; it reads a file and writes markdown.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

from experiment.authorization import (
    AuthorizationError, consume_authorization, cumulative, record_outcome,
    render_from_artifact,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCREEN = ROOT / "results/model-screen-v1/screen.json"


def _ledger(tmp_path) -> pathlib.Path:
    return tmp_path / "auth.json"


# ── Once-only ───────────────────────────────────────────────────────────────

def test_a_token_can_be_consumed_once(tmp_path):
    led = _ledger(tmp_path)
    consume_authorization("screen-v1", expected_requests=40, ledger=led)
    with pytest.raises(AuthorizationError) as err:
        consume_authorization("screen-v1", expected_requests=40, ledger=led)
    assert "already consumed" in str(err.value)
    assert "new authorization token" in str(err.value)


def test_an_empty_token_is_refused(tmp_path):
    with pytest.raises(AuthorizationError, match="no authorization token"):
        consume_authorization("", expected_requests=40, ledger=_ledger(tmp_path))


def test_a_different_token_is_allowed(tmp_path):
    """The guard blocks repetition, not the ability to authorize a NEW run."""
    led = _ledger(tmp_path)
    consume_authorization("run-a", expected_requests=40, ledger=led)
    consume_authorization("run-b", expected_requests=40, ledger=led)
    assert cumulative(led)["authorizations"] == 2


def test_the_claim_is_written_before_the_run_finishes(tmp_path):
    """A crashed run must still block a retry: the claim precedes execution."""
    led = _ledger(tmp_path)
    consume_authorization("screen-v1", expected_requests=40, ledger=led)
    state = json.loads(led.read_text())["screen-v1"]
    assert state["status"] == "in_progress"
    assert "finished_at" not in state
    # ...and a second attempt is refused while it is still in_progress.
    with pytest.raises(AuthorizationError):
        consume_authorization("screen-v1", expected_requests=40, ledger=led)


def test_outcome_records_requests_and_cost(tmp_path):
    led = _ledger(tmp_path)
    consume_authorization("s", expected_requests=40, ledger=led)
    entry = record_outcome("s", requests_sent=20, cost_usd=0.005143, ledger=led)
    assert entry["status"] == "completed"
    assert entry["requests_sent"] == 20
    assert entry["cost_usd"] == pytest.approx(0.005143)


def test_cumulative_totals_expose_an_aggregate_overrun(tmp_path):
    """The defect was two compliant runs summing over the ceiling."""
    led = _ledger(tmp_path)
    for token, req in (("authorized", 20), ("inadvertent", 21)):
        consume_authorization(token, expected_requests=40, ledger=led)
        record_outcome(token, requests_sent=req, cost_usd=0.001, ledger=led)
    total = cumulative(led)
    assert total["requests_sent"] == 41
    assert total["requests_sent"] > 40, "the aggregate must be visible as over"


def test_a_corrupt_ledger_fails_closed(tmp_path):
    led = _ledger(tmp_path)
    led.write_text("{not json")
    with pytest.raises(AuthorizationError, match="unreadable or corrupt"):
        consume_authorization("x", expected_requests=40, ledger=led)


# ── Render cannot dispatch ──────────────────────────────────────────────────

@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_render_reproduces_the_report_without_any_provider(tmp_path):
    out = tmp_path / "SCREEN.md"
    written = render_from_artifact(SCREEN, out)
    assert written.is_file()
    text = written.read_text()
    assert "Endpoint routing" in text
    assert "google/gemma-4-31b-it" in text


def test_render_takes_no_ask_callable_and_imports_no_provider():
    """Structural guarantee: rendering cannot reach the network.

    Parsed from source so a later 'convenience' parameter cannot quietly reintroduce
    an execution path into the render function.
    """
    src = (ROOT / "experiment/authorization.py").read_text()
    fn = next(node for node in ast.walk(ast.parse(src))
              if isinstance(node, ast.FunctionDef)
              and node.name == "render_from_artifact")
    params = [a.arg for a in fn.args.args]
    assert params == ["artifact", "out"], params

    body = ast.get_source_segment(src, fn)
    for forbidden in ("make_asker", "aiohttp", "urllib", "requests.post",
                      "provider_v2"):
        assert forbidden not in body, f"render must not reference {forbidden}"
    # It DOES use the pure report builder, which is fine.
    assert "render(" in body


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_render_fails_loudly_on_a_missing_or_empty_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_from_artifact(tmp_path / "nope.json", tmp_path / "o.md")
    empty = tmp_path / "empty.json"
    empty.write_text('{"rows": []}')
    with pytest.raises(ValueError, match="no rows"):
        render_from_artifact(empty, tmp_path / "o.md")


# ── The CLI refuses the shape that caused the incident ──────────────────────

def test_the_cli_requires_a_token_not_a_bare_flag():
    """`--authorized` alone must not be a valid invocation any more."""
    src = (ROOT / "experiment/model_screen.py").read_text()
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    opts = [a.value for n in ast.walk(main) if isinstance(n, ast.Call)
            for a in n.args if isinstance(a, ast.Constant)
            and isinstance(a.value, str) and a.value.startswith("--")]
    assert "--authorized" in opts
    assert "--render" in opts
    # --authorized now takes a TOKEN (metavar) rather than being store_true.
    body = ast.get_source_segment(src, main)
    assert 'metavar="TOKEN"' in body or "metavar='TOKEN'" in body
