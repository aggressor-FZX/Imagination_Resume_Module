
import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

# Load all real secrets from the primary workspace secret file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
shared_env_path = os.path.join(base_dir, ".env_shared_secrets")
load_dotenv(shared_env_path)

# Add the current directory to sys.path to import local modules
sys.path.append(os.getcwd())

from imaginator_new_integration import _resolve_onet_code, run_new_pipeline_async
from models import AnalysisRequest
from llm_client_adapter import LLMClientAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_onet_passthrough():
    logger.info("Testing O*NET Passthrough Logic (REAL API CHECK)...")
    
    # Test 1: Direct passthrough (Verification of upstream code)
    upstream_code = "15-1252.00"
    resolved = await _resolve_onet_code("Waitress", upstream_onet_code=upstream_code)
    
    assert resolved["code"] == upstream_code
    assert resolved["source"] == "upstream_passthrough"
    logger.info(f"Test 1 Passed: {resolved}")

    # Test 2: REAL O*NET Search Logic (No upstream code)
    # This hits the live O*NET API and LLM resolver
    
    # Initialize real LLM client for O*NET semantic mapping
    api_key_1 = os.getenv("OPENROUTER_API_KEY_1")
    api_key_2 = os.getenv("OPENROUTER_API_KEY_2")
    
    llm_client = LLMClientAdapter(
        api_key=api_key_1,
        fallback_keys=[api_key_2] if api_key_2 else []
    )
    
    logger.info("Triggering real O*NET/LLM resolution for 'Cloud Solutions Architect'...")
    resolved_real = await _resolve_onet_code("Cloud Solutions Architect", llm_client=llm_client)
    
    logger.info(f"Test 2 Resolved (Real API): {resolved_real}")
    assert "code" in resolved_real
    assert resolved_real["code"].startswith("15-") # Cloud/Software is usually 15-
    assert resolved_real["source"] in ["llm", "llm_validated", "keyword_scored"]

async def test_pipeline_real_integration():
    logger.info("Testing REAL Pipeline Integration (End-to-End Trace)...")
    
    resume_text = "Experienced Senior Software Engineer with 10 years of Python, AWS, and Kubernetes experience. Managed teams of 5 and delivered mission-critical fintech APIs."
    job_ad = "Looking for a Lead Software Engineer to refactor our legacy Go and Python microservices. Must have high-load experience and AWS expertise."
    
    api_keys = [os.getenv("OPENROUTER_API_KEY_1"), os.getenv("OPENROUTER_API_KEY_2")]
    api_keys = [k for k in api_keys if k]
    
    # Use real Golden Bullets for style transfer check
    golden_bullets = [
        "Reduced database latency by 45% by implementing a multi-layered Redis caching strategy for high-traffic endpoints.",
        "Architected and deployed a Kubernetes-based microservices mesh serving 2M+ active daily users with 99.99% uptime."
    ]

    try:
        logger.info("Running real pipeline async...")
        result = await run_new_pipeline_async(
            resume_text=resume_text,
            job_ad=job_ad,
            openrouter_api_keys=api_keys,
            onet_code="15-1252.00", # Software Developers
            golden_bullets=golden_bullets,
            projects=[{"name": "Open Source Caching Lab", "description": "Developed a custom LRU cache in Rust for low-latency data processing"}]
        )
        
        logger.info("Pipeline Output Keys: %s", result.keys())
        assert "final_written_section_markdown" in result
        assert len(result["final_written_section_markdown"]) > 100
        logger.info("Pipeline SUCCESS. Preview: %s...", result["final_written_section_markdown"][:200])
        
    except Exception as e:
        logger.exception("Pipeline failed under real conditions")
        raise e

if __name__ == "__main__":
    import sys
    # To save tokens/time, we can run selectively or all
    asyncio.run(test_onet_passthrough())
    asyncio.run(test_pipeline_real_integration())
