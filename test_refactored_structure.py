#!/usr/bin/env python3
"""
Test script to verify the refactored imaginator package structure.
This tests the modular architecture without making actual API calls.
"""
import sys
import os
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_imports():
    """Test that all modules can be imported correctly."""
    print("🔍 Testing imports...")
    
    try:
        from imaginator.config import settings, MODEL_STAGE_1, MODEL_STAGE_2, MODEL_STAGE_3, MODEL_STAGE_4
        print("✅ config.py imports successful")
        print(f"   Models: S1={MODEL_STAGE_1}, S2={MODEL_STAGE_2}, S3={MODEL_STAGE_3}, S4={MODEL_STAGE_4}")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        from imaginator.gateway import call_llm_async, get_model_cost
        print("✅ gateway.py imports successful")
    except Exception as e:
        print(f"❌ gateway.py import failed: {e}")
        return False
    
    try:
        from imaginator.microservices import (
            DocumentReader, FastSVM, Hermes, JobSearchAPI
        )
        print("✅ microservices.py imports successful")
    except Exception as e:
        print(f"❌ microservices.py import failed: {e}")
        return False
    
    try:
        from imaginator.orchestrator import run_full_funnel_pipeline
        print("✅ orchestrator.py imports successful")
    except Exception as e:
        print(f"❌ orchestrator.py import failed: {e}")
        return False
    
    try:
        from imaginator.stages.researcher import run_stage1_researcher
        from imaginator.stages.drafter import run_stage2_drafter
        from imaginator.stages.star_editor import run_stage3_star_editor
        from imaginator.stages.polisher import run_stage4_polisher
        print("✅ All stage modules import successful")
    except Exception as e:
        print(f"❌ Stage imports failed: {e}")
        return False
    
    return True


async def test_stage_1_researcher():
    """Test Stage 1: Researcher with mock data."""
    print("\n🧪 Testing Stage 1: Researcher...")
    
    from imaginator.stages.researcher import run_stage1_researcher
    
    # Mock data
    resume_text = "Senior Python Developer with 5 years experience"
    job_ad = "Looking for Python developer with FastAPI and PostgreSQL"
    hermes_data = {"skills": ["Python", "FastAPI"], "experience": "5 years"}
    svm_data = {"validated_skills": ["Python", "FastAPI"], "confidence": 0.95}
    
    # Mock the gateway function
    with patch('imaginator.stages.researcher.call_llm_async') as mock_call:
        mock_call.return_value = json.dumps({
            "master_profile": "Experienced Python developer...",
            "industry_intel": "FastAPI is trending...",
            "tailoring_strategy": "Focus on backend skills",
            "key_metrics_to_use": ["5 years", "95% confidence"]
        })
        
        result = await run_stage1_researcher(resume_text, job_ad, hermes_data, svm_data)
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "master_profile" in result, "Should contain master_profile"
        assert "industry_intel" in result, "Should contain industry_intel"
        print("✅ Stage 1 test passed")
        return True


async def test_stage_2_drafter():
    """Test Stage 2: Drafter with mock data."""
    print("\n🧪 Testing Stage 2: Drafter...")
    
    from imaginator.stages.drafter import run_stage2_drafter
    
    research_data = {
        "master_profile": "Experienced Python developer...",
        "tailoring_strategy": "Focus on backend skills",
        "key_metrics_to_use": ["5 years"]
    }
    
    with patch('imaginator.stages.drafter.call_llm_async') as mock_call:
        mock_call.return_value = "Creative draft with narrative flow..."
        
        result = await run_stage2_drafter(research_data)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        print("✅ Stage 2 test passed")
        return True


async def test_stage_3_star_editor():
    """Test Stage 3: STAR Editor with mock data."""
    print("\n🧪 Testing Stage 3: STAR Editor...")
    
    from imaginator.stages.star_editor import run_stage3_star_editor
    
    creative_draft = "Built amazing APIs that improved performance..."
    research_data = {
        "star_suggestions": [
            {
                "experience_index": 0,
                "result_metrics": ["60% faster", "99.9% uptime"]
            }
        ]
    }
    experiences = [{"role": "Senior Developer", "duration": "5 years"}]
    
    with patch('imaginator.stages.star_editor.call_llm_async') as mock_call:
        mock_call.return_value = "# Professional Experience\n\n- **Situation**: Slow deployment\n- **Task**: Improve speed\n- **Action**: Implemented CI/CD\n- **Result**: 60% faster"
        
        result = await run_stage3_star_editor(creative_draft, research_data, experiences)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        print("✅ Stage 3 test passed")
        return True


async def test_stage_4_polisher():
    """Test Stage 4: Polisher with mock data."""
    print("\n🧪 Testing Stage 4: Polisher...")
    
    from imaginator.stages.polisher import run_stage4_polisher
    
    star_draft = "# Experience\n\n- Built APIs..."
    original_job_ad = "Python developer needed for FastAPI work"
    
    with patch('imaginator.stages.polisher.call_llm_async') as mock_call:
        mock_call.return_value = json.dumps({
            "final_resume_markdown": "# Final Resume\n\n- Built APIs...",
            "editorial_notes": "QC passed, keywords aligned"
        })
        
        result = await run_stage4_polisher(star_draft, original_job_ad)
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "final_resume_markdown" in result, "Should contain final_resume_markdown"
        assert "editorial_notes" in result, "Should contain editorial_notes"
        print("✅ Stage 4 test passed")
        return True


async def test_orchestrator_flow():
    """Test the full orchestrator flow with mocks."""
    print("\n🧪 Testing Orchestrator Flow...")
    
    from imaginator.orchestrator import run_full_funnel_pipeline
    
    # Mock all stage functions
    with patch('imaginator.orchestrator.run_stage1_researcher') as mock_s1, \
         patch('imaginator.orchestrator.run_stage2_drafter') as mock_s2, \
         patch('imaginator.orchestrator.run_stage3_star_editor') as mock_s3, \
         patch('imaginator.orchestrator.run_stage4_polisher') as mock_s4:
        
        # Setup mock returns
        mock_s1.return_value = {"master_profile": "Dossier compiled"}
        mock_s2.return_value = "Creative draft"
        mock_s3.return_value = "STAR formatted"
        mock_s4.return_value = {
            "final_resume_markdown": "Final resume",
            "editorial_notes": "QC passed"
        }
        
        # Test data
        resume_text = "Python developer"
        job_ad = "Python job"
        hermes_data = {"skills": ["Python"]}
        svm_data = {"validated_skills": ["Python"]}
        
        result = await run_full_funnel_pipeline(resume_text, job_ad, hermes_data, svm_data)
        
        # Verify flow
        assert mock_s1.called, "Stage 1 should be called"
        assert mock_s2.called, "Stage 2 should be called"
        assert mock_s3.called, "Stage 3 should be called"
        assert mock_s4.called, "Stage 4 should be called"
        
        # Verify data flow (Stage 1 gets everything, later stages get less)
        s1_call = mock_s1.call_args[0]
        assert s1_call[0] == resume_text, "Stage 1 gets resume_text"
        assert s1_call[1] == job_ad, "Stage 1 gets job_ad"
        assert s1_call[2] == hermes_data, "Stage 1 gets hermes_data"
        assert s1_call[3] == svm_data, "Stage 1 gets svm_data"
        
        # Stage 2 only gets research data
        s2_call = mock_s2.call_args[0]
        assert s2_call[0] == {"master_profile": "Dossier compiled"}, "Stage 2 gets research data"
        
        # Stage 4 re-injects job_ad
        s4_call = mock_s4.call_args[0]
        assert s4_call[1] == job_ad, "Stage 4 gets original job_ad"
        
        print("✅ Orchestrator flow test passed")
        print("   ✅ Heavy Start: Stage 1 gets all data")
        print("   ✅ Lean Middle: Stages 2-3 get filtered data")
        print("   ✅ Analytical Finish: Stage 4 re-injects job_ad")
        return True


async def test_microservices():
    """Test microservices connectors."""
    print("\n🧪 Testing Microservices...")
    
    from imaginator.microservices import DocumentReader, FastSVM, Hermes, JobSearchAPI
    
    # Test DocumentReader
    with patch('imaginator.microservices.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"extraction": {"raw_text": "Extracted text"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        reader = DocumentReader("test_key")
        result = reader.process_file("test.pdf")
        assert result["extraction"]["raw_text"] == "Extracted text"
        print("✅ DocumentReader connector works")
    
    # Test FastSVM
    with patch('imaginator.microservices.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"validated_skills": ["Python"]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        svm = FastSVM("test_key")
        result = svm.process_resume("Python developer")
        assert result["validated_skills"] == ["Python"]
        print("✅ FastSVM connector works")
    
    # Test Hermes
    with patch('imaginator.microservices.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"skills": ["Python"], "insights": "Great"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        hermes = Hermes("test_key")
        result = hermes.extract("Python developer")
        assert result["skills"] == ["Python"]
        print("✅ Hermes connector works")
    
    # Test JobSearchAPI
    with patch('imaginator.microservices.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"jobs": [{"title": "Python Dev"}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        job_api = JobSearchAPI("test_key")
        result = job_api.search("Python", "Remote")
        assert result["jobs"][0]["title"] == "Python Dev"
        print("✅ JobSearchAPI connector works")
    
    return True


async def test_gateway_cost_estimation():
    """Test gateway cost estimation."""
    print("\n🧪 Testing Gateway Cost Estimation...")
    
    from imaginator.gateway import get_model_cost
    
    # Test cost calculation
    cost = get_model_cost("deepseek/deepseek-v3.2:online", 1000, 500)
    assert cost > 0, "Cost should be positive"
    print(f"✅ Cost estimation works: ${cost:.4f} for 1000 input + 500 output tokens")
    
    # Test unknown model
    cost_unknown = get_model_cost("unknown/model", 1000, 500)
    assert cost_unknown == 0.0, "Unknown model should return 0"
    print("✅ Unknown model handling works")
    
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING REFACTORED IMAGINATOR STRUCTURE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Stage 1: Researcher", test_stage_1_researcher),
        ("Stage 2: Drafter", test_stage_2_drafter),
        ("Stage 3: STAR Editor", test_stage_3_star_editor),
        ("Stage 4: Polisher", test_stage_4_polisher),
        ("Orchestrator Flow", test_orchestrator_flow),
        ("Microservices", test_microservices),
        ("Gateway Cost Estimation", test_gateway_cost_estimation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"❌ {name} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The refactored structure is working correctly.")
        print("\n📦 Package Structure Summary:")
        print("   imaginator/")
        print("   ├── config.py           ✅ Centralized configuration")
        print("   ├── gateway.py          ✅ LLM logic & cost tracking")
        print("   ├── microservices.py    ✅ External service connectors")
        print("   ├── orchestrator.py     ✅ 4-stage funnel pipeline")
        print("   └── stages/")
        print("       ├── researcher.py   ✅ Stage 1: Heavy Start")
        print("       ├── drafter.py      ✅ Stage 2: Creative Draft")
        print("       ├── star_editor.py  ✅ Stage 3: STAR Formatting")
        print("       └── polisher.py     ✅ Stage 4: Analytical Finish")
        return True
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)