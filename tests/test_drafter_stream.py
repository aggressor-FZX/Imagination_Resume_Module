import json
import pytest


@pytest.mark.asyncio
async def test_drafter_stream_assembles_valid_json(monkeypatch):
    from stages.drafter import Drafter

    chunks = [
        '{"rewritten_experiences":[',
        '{"company":"Acme","title":"Analyst","bullets":["Built dashboards"]}',
        '],"total_bullets":1}',
    ]

    class FakeClient:
        async def call_llm_async(self, **kwargs):
            on_token = kwargs.get("on_token")
            # Prefer stream path if draft wires stream=True
            text = "".join(chunks)
            if on_token:
                for c in chunks:
                    on_token(c)
            return text

    d = Drafter(FakeClient())
    seen = []
    out = await d.draft(
        experiences=[{"company": "Acme", "title": "Analyst", "bullets": ["dashboards"]}],
        job_ad="Data Analyst",
        research_data={"implied_metrics": [], "domain_vocab": [], "implied_skills": []},
        on_token=seen.append,
    )
    assert seen, "expected token callbacks"
    assert out.get("rewritten_experiences")
    assert out["rewritten_experiences"][0]["company"] == "Acme"
