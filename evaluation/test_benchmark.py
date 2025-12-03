#!/usr/bin/env python3
"""
Quick test script to verify benchmark setup
"""

import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "evaluation"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from evaluation.compensation_benchmark import (
            CompensationBenchmark,
            ExperimentConfig,
            FailureMode
        )
        print("✓ compensation_benchmark imported")
    except Exception as e:
        print(f"✗ compensation_benchmark import failed: {e}")
        return False
    
    try:
        from evaluation.enterprise_workflows import (
            ENTERPRISE_TASKS,
            ENTERPRISE_COMPENSATION_MAPPING
        )
        print("✓ enterprise_workflows imported")
    except Exception as e:
        print(f"✗ enterprise_workflows import failed: {e}")
        return False
    
    try:
        from evaluation.benchmark_analysis import BenchmarkAnalyzer
        print("✓ benchmark_analysis imported")
    except Exception as e:
        print(f"✗ benchmark_analysis import failed: {e}")
        return False
    
    return True


def test_benchmark_init():
    """Test benchmark initialization"""
    print("\nTesting benchmark initialization...")
    
    try:
        from evaluation.compensation_benchmark import CompensationBenchmark
        
        benchmark = CompensationBenchmark(output_dir="test_results")
        print(f"✓ Benchmark initialized with {len(benchmark.runners)} runners")
        print(f"  Available runners: {list(benchmark.runners.keys())}")
        return True
    except Exception as e:
        print(f"✗ Benchmark initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_config():
    """Test experiment configuration"""
    print("\nTesting experiment configuration...")
    
    try:
        from evaluation.compensation_benchmark import ExperimentConfig, FailureMode
        
        config = ExperimentConfig(
            task_id="P1",
            framework="compensation_lib",
            failure_rate=0.25,
            failure_mode=FailureMode.TOOL_EXECUTION_ERROR,
            num_runs=5
        )
        
        config_dict = config.to_dict()
        print(f"✓ Experiment config created: {config_dict}")
        return True
    except Exception as e:
        print(f"✗ Experiment config failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enterprise_workflows():
    """Test enterprise workflow definitions"""
    print("\nTesting enterprise workflows...")
    
    try:
        from evaluation.enterprise_workflows import ENTERPRISE_TASKS, ENTERPRISE_COMPENSATION_MAPPING
        
        print(f"✓ Found {len(ENTERPRISE_TASKS)} enterprise tasks:")
        for task_id, task_def in ENTERPRISE_TASKS.items():
            print(f"  - {task_id}: {task_def.name}")
        
        print(f"✓ Found {len(ENTERPRISE_COMPENSATION_MAPPING)} compensation mappings")
        return True
    except Exception as e:
        print(f"✗ Enterprise workflows test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Compensation Benchmark Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_benchmark_init,
        test_experiment_config,
        test_enterprise_workflows,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed! Benchmark is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
