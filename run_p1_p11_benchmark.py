#!/usr/bin/env python3
"""
REALM-Bench P1-P11 Benchmark Runner

Compares langchain-compensation vs SagaLLM on all 11 planning tasks.

Usage:
    python run_p1_p11_benchmark.py                    # Run all tasks
    python run_p1_p11_benchmark.py --tasks P1 P5 P11  # Run specific tasks
    python run_p1_p11_benchmark.py --runs 3           # Multiple runs per task
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "evaluation"))

from dotenv import load_dotenv
load_dotenv()

from evaluation.task_definitions import TASK_DEFINITIONS
from evaluation.compensation_runner_lib import CompensationLibRunner
from evaluation.saga_llm_runner import SagaLLMRunner, SAGA_AVAILABLE

# All P1-P11 tasks
ALL_TASKS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"]


def run_benchmark(
    tasks: List[str],
    frameworks: List[str],
    num_runs: int = 1,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """Run benchmark on specified tasks and frameworks."""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize runners
    runners = {}
    if "compensation_lib" in frameworks:
        try:
            runners["compensation_lib"] = CompensationLibRunner()
            print("✓ CompensationLibRunner initialized")
        except Exception as e:
            print(f"✗ CompensationLibRunner failed: {e}")

    if "sagallm" in frameworks and SAGA_AVAILABLE:
        try:
            runners["sagallm"] = SagaLLMRunner()
            print("✓ SagaLLMRunner initialized")
        except Exception as e:
            print(f"✗ SagaLLMRunner failed: {e}")

    if not runners:
        print("ERROR: No runners available!")
        return {}

    # Results storage
    all_results = {fw: [] for fw in runners.keys()}
    summary = {
        "timestamp": timestamp,
        "tasks": tasks,
        "frameworks": list(runners.keys()),
        "num_runs": num_runs,
        "results": {}
    }

    # Run benchmarks
    total_runs = len(tasks) * len(runners) * num_runs
    current_run = 0

    print(f"\n{'='*60}")
    print(f"REALM-Bench P1-P11 Comparison")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Frameworks: {', '.join(runners.keys())}")
    print(f"Runs per task: {num_runs}")
    print(f"Total runs: {total_runs}")
    print(f"{'='*60}\n")

    for task_id in tasks:
        if task_id not in TASK_DEFINITIONS:
            print(f"⚠ Task {task_id} not found, skipping")
            continue

        task_def = TASK_DEFINITIONS[task_id]
        print(f"\n{'─'*40}")
        print(f"Task: {task_id} - {task_def.name}")
        print(f"{'─'*40}")

        for framework, runner in runners.items():
            for run_num in range(num_runs):
                current_run += 1
                print(f"  [{current_run}/{total_runs}] {framework} run {run_num + 1}...", end=" ", flush=True)

                try:
                    start = time.time()
                    result = runner(task_def)
                    elapsed = time.time() - start

                    result["task_id"] = task_id
                    result["framework"] = framework
                    result["run_number"] = run_num + 1
                    result["execution_time"] = elapsed

                    all_results[framework].append(result)

                    goal_rate = result.get("metrics", {}).get("goal_satisfaction_rate", 0)
                    print(f"✓ {elapsed:.2f}s (goals: {goal_rate*100:.0f}%)")

                except Exception as e:
                    print(f"✗ Error: {e}")
                    all_results[framework].append({
                        "task_id": task_id,
                        "framework": framework,
                        "run_number": run_num + 1,
                        "error": str(e)
                    })

    # Calculate summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for framework, results in all_results.items():
        valid_results = [r for r in results if "error" not in r]

        if valid_results:
            avg_time = sum(r.get("execution_time", 0) for r in valid_results) / len(valid_results)
            avg_goals = sum(r.get("metrics", {}).get("goal_satisfaction_rate", 0) for r in valid_results) / len(valid_results)

            summary["results"][framework] = {
                "total_runs": len(results),
                "successful_runs": len(valid_results),
                "failed_runs": len(results) - len(valid_results),
                "avg_execution_time": avg_time,
                "avg_goal_satisfaction": avg_goals,
            }

            print(f"\n{framework}:")
            print(f"  Successful runs: {len(valid_results)}/{len(results)}")
            print(f"  Avg execution time: {avg_time:.2f}s")
            print(f"  Avg goal satisfaction: {avg_goals*100:.1f}%")

    # Save results
    for framework, results in all_results.items():
        filepath = os.path.join(output_dir, f"{framework}_results_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Saved {framework} results to {filepath}")

    summary_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")

    # Generate comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Task':<8} {'Framework':<20} {'Time':<10} {'Goals':<10}")
    print("-" * 50)

    for task_id in tasks:
        for framework in runners.keys():
            task_results = [r for r in all_results[framework]
                          if r.get("task_id") == task_id and "error" not in r]
            if task_results:
                avg_time = sum(r.get("execution_time", 0) for r in task_results) / len(task_results)
                avg_goals = sum(r.get("metrics", {}).get("goal_satisfaction_rate", 0) for r in task_results) / len(task_results)
                print(f"{task_id:<8} {framework:<20} {avg_time:<10.2f} {avg_goals*100:<10.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="REALM-Bench P1-P11 Benchmark")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        help="Tasks to run (default: all P1-P11)")
    parser.add_argument("--frameworks", nargs="+",
                        default=["compensation_lib", "sagallm"],
                        help="Frameworks to benchmark")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per task (default: 1)")
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Output directory for results")

    args = parser.parse_args()

    # Validate tasks
    valid_tasks = [t for t in args.tasks if t in ALL_TASKS]
    if not valid_tasks:
        print(f"ERROR: No valid tasks specified. Valid tasks: {ALL_TASKS}")
        sys.exit(1)

    run_benchmark(
        tasks=valid_tasks,
        frameworks=args.frameworks,
        num_runs=args.runs,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
