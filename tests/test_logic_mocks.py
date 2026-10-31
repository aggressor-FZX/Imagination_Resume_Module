import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock
from stages.researcher import Researcher
from stages.drafter import Drafter
from orchestrator import PipelineOrchestrator

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.call_llm_async = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_researcher_stage_logic(mock_llm_client):
    researcher = Researcher(mock_llm_client)
    mock_llm_client.call_llm_async.return_value = json.dumps({
        "work_archetypes": ["Scaling"],
        "implied_metrics": ["1M+ users"],
        "domain_vocab": ["Kubernetes"],
        "implied_skills": ["Distributed Systems"]
    })
    
    result = await researcher.analyze("Job Ad", [{"role": "Eng", "company": "Tech"}])
    assert "Scaling" in result["work_archetypes"]
    assert "1M+ users" in result["implied_metrics"]

@pytest.mark.asyncio
async def test_drafter_quality_gate_retry(mock_llm_client):
    """
    Test that the orchestrator retries if the drafter returns placeholders
    or low quantification.
    """
    orchestrator = PipelineOrchestrator(mock_llm_client)
    
    # First response: Has placeholder
    bad_response = json.dumps({
        "rewritten_experiences": [{
            "company": "Tech",
            "role": "Eng",
            "bullets": ["Worked on [PLACEHOLDER] project"],
            "metrics_used": []
        }],
        "seniority_applied": "mid"
    })
    
    # Second response: Good
    good_response = json.dumps({
        "rewritten_experiences": [{
            "company": "Tech",
            "role": "Eng",
            "bullets": ["Increased efficiency by 25% using Python"],
            "metrics_used": ["25%"]
        }],
        "seniority_applied": "mid"
    })
    
    mock_llm_client.call_llm_async.side_effect = [
        # Researcher call
        json.dumps({"implied_metrics": [], "domain_vocab": [], "work_archetypes": []}),
        # Drafter attempt 1 (Bad)
        bad_response,
        # Drafter attempt 2 (Good)
        good_response,
        # Star editor call
        json.dumps({"final_markdown": "Resume Content", "editor_notes": ""})
    ]
    
    result = await orchestrator.run_pipeline(
        resume_text="Resume",
        job_ad="Job",
        experiences=[{"company": "Tech", "role": "Eng"}],
        creativity_mode="balanced"
    )
    
    # Verify call count: 1 (researcher) + 1 (drafter) + 1 (editor) + metadata
    assert mock_llm_client.call_llm_async.call_count >= 3

    # Phase 2 (AR-2) — BEHAVIOUR CHANGE, DOCUMENTED.
    #
    # This test was named "quality_gate_retry" and expected the orchestrator to
    # discard a placeholder-bearing draft and re-call the drafter. That retry
    # was deliberately REMOVED in orchestrator.py:272 ("single attempt only to
    # prevent OOM from retry holding memory") because a second drafter call
    # OOM-killed the 1 GB Render instance.
    #
    # So the assertion was testing behaviour that no longer exists — it could
    # never pass again. Rather than re-enable a known OOM path or silently
    # delete the test, it now asserts what the pipeline actually does:
    # a single draft attempt, whose output is cleaned downstream by the
    # editor's placeholder guard rather than by a drafter retry.
    final_output = result.get("final_output") or {}
    assert final_output, "pipeline produced no final_output"
    markdown = final_output.get("final_written_section_markdown") or ""
    assert markdown.strip(), "editor stage produced empty markdown"
    assert mock_llm_client.call_llm_async.call_count <= 4, (
        "more than one drafter attempt observed; the single-attempt OOM guard "
        "in orchestrator.py:272 may have been removed"
    )

@pytest.mark.asyncio
@pytest.mark.live
async def test_onet_passthrough_logic(mock_llm_client, live_credentials):
    """
    Verify that if onet_code is provided, we don't hit the LLM resolver for it.
    The integration layer constructs its own real client; the mock fixture does
    not intercept it, so this is a live test rather than an offline unit test.
    """
    from imaginator_new_integration import run_new_pipeline_async
    
    mock_llm_client.call_llm_async.return_value = json.dumps({"score": 0.9})
    
    # We provide a valid O*NET code
    await run_new_pipeline_async(
        resume_text="Resume",
        job_ad="Job",
        onet_code="15-1252.00",
        openrouter_api_keys=[
            live_credentials[name]
            for name in ("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2")
            if live_credentials[name]
        ],
    )
