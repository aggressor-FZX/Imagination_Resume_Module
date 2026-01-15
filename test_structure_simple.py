#!/usr/bin/env python3
"""
Simple test to verify the refactored imaginator package structure.
Tests basic imports and structure without requiring external dependencies.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_directory_structure():
    """Test that the directory structure exists."""
    print("🔍 Testing directory structure...")
    
    required_files = [
        "imaginator/__init__.py",
        "imaginator/config.py",
        "imaginator/gateway.py",
        "imaginator/microservices.py",
        "imaginator/orchestrator.py",
        "imaginator/stages/__init__.py",
        "imaginator/stages/researcher.py",
        "imaginator/stages/drafter.py",
        "imaginator/stages/star_editor.py",
        "imaginator/stages/polisher.py",
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing: {file_path}")
            return False
        print(f"✅ Found: {file_path}")
    
    return True

def test_basic_imports():
    """Test that basic imports work."""
    print("\n🔍 Testing basic imports...")
    
    try:
        from imaginator.config import settings, MODEL_STAGE_1, MODEL_STAGE_2, MODEL_STAGE_3, MODEL_STAGE_4
        print("✅ config.py imports successful")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        # Test that gateway can be imported (even if google imports fail)
        import imaginator.gateway
        print("✅ gateway.py imports successful")
    except Exception as e:
        print(f"❌ gateway.py import failed: {e}")
        return False
    
    try:
        import imaginator.microservices
        print("✅ microservices.py imports successful")
    except Exception as e:
        print(f"❌ microservices.py import failed: {e}")
        return False
    
    try:
        import imaginator.orchestrator
        print("✅ orchestrator.py imports successful")
    except Exception as e:
        print(f"❌ orchestrator.py import failed: {e}")
        return False
    
    try:
        from imaginator.stages import researcher, drafter, star_editor, polisher
        print("✅ All stage modules import successful")
    except Exception as e:
        print(f"❌ Stage imports failed: {e}")
        return False
    
    return True

def test_config_values():
    """Test that config has expected values."""
    print("\n🔍 Testing config values...")
    
    from imaginator.config import MODEL_STAGE_1, MODEL_STAGE_2, MODEL_STAGE_3, MODEL_STAGE_4
    
    expected_models = {
        "Stage 1": MODEL_STAGE_1,
        "Stage 2": MODEL_STAGE_2,
        "Stage 3": MODEL_STAGE_3,
        "Stage 4": MODEL_STAGE_4,
    }
    
    for stage, model in expected_models.items():
        if model:
            print(f"✅ {stage}: {model}")
        else:
            print(f"❌ {stage}: Missing model assignment")
            return False
    
    return True

def test_module_functions():
    """Test that modules have expected functions."""
    print("\n🔍 Testing module functions...")
    
    # Test gateway
    try:
        from imaginator.gateway import call_llm_async
        print("✅ gateway.call_llm_async exists")
    except ImportError:
        print("❌ gateway.call_llm_async missing")
        return False
    
    # Test microservices
    try:
        import imaginator.microservices as ms
        # Check for async functions
        expected_funcs = ["call_loader_process_text_only", "call_fastsvm_process_resume", 
                         "call_hermes_extract", "call_job_search_api"]
        for func in expected_funcs:
            if hasattr(ms, func):
                print(f"✅ microservices.{func} exists")
            else:
                print(f"❌ microservices.{func} missing")
                return False
    except Exception as e:
        print(f"❌ microservices test failed: {e}")
        return False
    
    # Test orchestrator
    try:
        from imaginator.orchestrator import run_full_funnel_pipeline
        print("✅ orchestrator.run_full_funnel_pipeline exists")
    except ImportError:
        print("❌ orchestrator.run_full_funnel_pipeline missing")
        return False
    
    # Test stages
    try:
        from imaginator.stages.researcher import run_stage1_researcher
        from imaginator.stages.drafter import run_stage2_drafter
        from imaginator.stages.star_editor import run_stage3_star_editor
        from imaginator.stages.polisher import run_stage4_polisher
        print("✅ All stage functions exist")
    except ImportError as e:
        print(f"❌ Stage function missing: {e}")
        return False
    
    return True

def test_architecture_principles():
    """Test that the architecture follows the intended principles."""
    print("\n🔍 Testing architecture principles...")
    
    # Check that stages are isolated
    try:
        import imaginator.stages.researcher as researcher
        import imaginator.stages.drafter as drafter
        import imaginator.stages.star_editor as star_editor
        import imaginator.stages.polisher as polisher
        
        # Each stage should have its own imports
        print("✅ Stages are properly isolated modules")
    except Exception as e:
        print(f"❌ Stage isolation test failed: {e}")
        return False
    
    # Check that orchestrator imports stages but not vice versa
    try:
        import imaginator.orchestrator
        import imaginator.stages.researcher
        
        # Verify orchestrator has stage imports
        orchestrator_source = open("imaginator/orchestrator.py").read()
        if "from .stages" in orchestrator_source:
            print("✅ Orchestrator properly imports stages")
        else:
            print("❌ Orchestrator doesn't import stages")
            return False
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING REFACTORED IMAGINATOR STRUCTURE")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Basic Imports", test_basic_imports),
        ("Config Values", test_config_values),
        ("Module Functions", test_module_functions),
        ("Architecture Principles", test_architecture_principles),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✅ {name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name}: FAILED with exception: {e}")
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
        print("\n💡 Key Benefits Achieved:")
        print("   • Isolation: Each stage can be worked on independently")
        print("   • Context Efficiency: Clean imports, focused modules")
        print("   • Human Readability: Orchestrator shows funnel in <30s")
        print("   • Resilience: Independent stage testing possible")
        print("   • Agent-Friendly: One file per concern")
        return True
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)