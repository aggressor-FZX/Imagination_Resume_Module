import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY_1", "test-key-1")
os.environ.setdefault("OPENROUTER_API_KEY_2", "test-key-2")
os.environ.setdefault("IMAGINATOR_AUTH_TOKEN", "test-token")

import app as app_module  # noqa: E402
from app import app, get_api_key  # noqa: E402


def _parse_sse(text: str):
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name, data = "message", None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        events.append((name, data))
    return events


@pytest.fixture
def client(monkeypatch):
    app.dependency_overrides[get_api_key] = lambda: "test"
    monkeypatch.setattr(app_module.settings, "openrouter_api_key_1", "k", raising=False)

    class FakeResearcher:
        def __init__(self, _client):
            pass

        async def analyze(self, job_ad, experiences=None):
            return {
                "implied_metrics": ["m1"],
                "domain_vocab": ["v1", "v2"],
                "implied_skills": ["s1"],
                "work_archetypes": [],
            }

    monkeypatch.setattr(app_module, "Researcher", FakeResearcher)
    monkeypatch.setattr(app_module, "LLMClientAdapter", lambda **kw: object())
    yield TestClient(app)
    app.dependency_overrides.pop(get_api_key, None)


def test_research_json_unchanged(client):
    r = client.post("/research", json={"job_ad": "Data Analyst"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["research_data"]["domain_vocab"] == ["v1", "v2"]
    assert body["forked"] is True


def test_research_sse_stage_partial_result(client):
    r = client.post(
        "/research",
        json={"job_ad": "Data Analyst"},
        headers={"Accept": "text/event-stream"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "stage"
    assert "research_partial" in names
    assert names[-1] == "result"
    partial = dict(events)["research_partial"]
    assert partial["domain_vocab_preview"] == ["v1", "v2"]
    json_body = client.post("/research", json={"job_ad": "Data Analyst"}).json()
    sse_body = events[-1][1]
    json_body.pop("duration_ms")
    sse_body.pop("duration_ms")
    assert sse_body == json_body


def test_research_returns_researcher_run_metrics(client, monkeypatch):
    class FakeAdapter:
        def __init__(self, **_kw):
            pass

        def get_usage_stats(self):
            return {
                "calls": [{"model": "m", "prompt_tokens": 12, "completion_tokens": 3}],
                "total_prompt_tokens": 12,
                "total_completion_tokens": 3,
            }

    monkeypatch.setattr(app_module, "LLMClientAdapter", FakeAdapter)
    body = client.post("/research", json={"job_ad": "Data Analyst"}).json()
    assert body["status"] == "completed"
    assert body["run_metrics"]["calls"] == [
        {"model": "m", "prompt_tokens": 12, "completion_tokens": 3, "stage": "researcher"}
    ]


def test_research_run_metrics_failure_never_breaks_research(client):
    body = client.post("/research", json={"job_ad": "Data Analyst"}).json()
    assert body["status"] == "completed"
    assert body["run_metrics"] is None


def test_research_sse_timeout_falls_back(client, monkeypatch):
    import asyncio

    class SlowResearcher:
        def __init__(self, _client):
            pass

        async def analyze(self, job_ad, experiences=None):
            await asyncio.sleep(5)

    monkeypatch.setattr(app_module, "Researcher", SlowResearcher)
    monkeypatch.setitem(app_module.TIMEOUTS, "researcher", 0.2)
    r = client.post(
        "/research",
        json={"job_ad": "Data Analyst"},
        headers={"Accept": "text/event-stream"},
    )
    events = _parse_sse(r.text)
    assert events[-1][0] == "result"
    assert events[-1][1]["status"] == "timeout"
    assert events[-1][1]["research_data"]["fallback"] is True
    assert any(n == "stage" and d.get("state") == "fallback" for n, d in events)
