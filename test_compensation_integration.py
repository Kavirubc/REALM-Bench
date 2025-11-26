#!/usr/bin/env python3
"""
Test script to verify compensation integration with REALM-Bench
"""

import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    
    try:
        from evaluation.compensation_runner import CompensationLangGraphRunner
        print("✓ CompensationLangGraphRunner imported successfully")
    except Exception as e:
        print(f"✗ Failed to import CompensationLangGraphRunner: {e}")
        return False
    
    try:
        from evaluation.compensation_metrics import CompensationMetrics
        print("✓ CompensationMetrics imported successfully")
    except Exception as e:
        print(f"✗ Failed to import CompensationMetrics: {e}")
        return False
    
    try:
        from evaluation.compensation_tasks import COMPENSATION_TASK_DEFINITIONS
        print(f"✓ Compensation tasks imported successfully ({len(COMPENSATION_TASK_DEFINITIONS)} tasks)")
    except Exception as e:
        print(f"✗ Failed to import compensation tasks: {e}")
        return False
    
    try:
        from evaluation.framework_runners import get_framework_runners
        runners = get_framework_runners()
        if 'compensation' in runners:
            print("✓ Compensation runner registered in framework runners")
        else:
            print("✗ Compensation runner not found in framework runners")
            return False
    except Exception as e:
        print(f"✗ Failed to get framework runners: {e}")
        return False
    
    try:
        from evaluation.task_definitions import TASK_DEFINITIONS
        comp_tasks = [k for k in TASK_DEFINITIONS.keys() if k.startswith('CT')]
        if comp_tasks:
            print(f"✓ Compensation tasks found in TASK_DEFINITIONS: {comp_tasks}")
        else:
            print("✗ No compensation tasks found in TASK_DEFINITIONS")
            return False
    except Exception as e:
        print(f"✗ Failed to check TASK_DEFINITIONS: {e}")
        return False
    
    return True


def test_runner_initialization():
    """Test that the compensation runner can be initialized"""
    print("\nTesting runner initialization...")
    
    try:
        from evaluation.compensation_runner import CompensationLangGraphRunner
        runner = CompensationLangGraphRunner()
        print("✓ CompensationLangGraphRunner initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize CompensationLangGraphRunner: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Compensation Integration Test")
    print("=" * 60)
    
    if not test_imports():
        print("\n❌ Import tests failed")
        return 1
    
    if not test_runner_initialization():
        print("\n❌ Runner initialization test failed")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nYou can now run evaluations with:")
    print("  python run_evaluation.py --frameworks compensation --tasks CT1,CT2,CT3")
    print("  python run_evaluation.py --frameworks compensation,langgraph --tasks P4,P7,P8,P9")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

