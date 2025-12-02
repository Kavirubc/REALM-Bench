#!/usr/bin/env python3
"""
Benchmark langchain-compensation vs SagaLLM on ACID transaction tasks.

This script compares:
- sagallm: SagaLLM multi-agent framework
- compensation_lib: langchain-compensation single-agent
- compensation_multiagent: langchain-compensation multi-agent

Usage:
    python benchmark_compensation.py
    python benchmark_compensation.py --frameworks sagallm,compensation_lib --tasks P5-ACID --runs 5
    python benchmark_compensation.py --output results.json
"""
import argparse
import json
import time
import logging
import os
import sys
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from tabulate import tabulate
from evaluation.framework_runners import get_framework_runners
from evaluation.compensation_tasks import COMPENSATION_TASK_DEFINITIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def run_benchmark(
    frameworks: List[str],
    tasks: List[str],
    runs: int
) -> Dict[str, Any]:
    """
    Run benchmark for specified frameworks and tasks.

    Args:
        frameworks: List of framework names to benchmark
        tasks: List of task IDs to run
        runs: Number of runs per task per framework

    Returns:
        Dictionary containing all run results and summary statistics
    """
    runners = get_framework_runners()
    results = {
        "config": {
            "frameworks": frameworks,
            "tasks": tasks,
            "runs_per_task": runs,
        },
        "runs": [],
        "summary": {}
    }

    total_runs = len(frameworks) * len(tasks) * runs
    current_run = 0

    for task_id in tasks:
        task_def = COMPENSATION_TASK_DEFINITIONS.get(task_id)
        if not task_def:
            logger.warning(f"Task {task_id} not found in COMPENSATION_TASK_DEFINITIONS")
            continue

        for framework in frameworks:
            runner = runners.get(framework)
            if not runner:
                logger.warning(f"Framework {framework} not available")
                continue

            for run_num in range(runs):
                current_run += 1
                logger.info(f"[{current_run}/{total_runs}] Running {framework} on {task_id} (run {run_num + 1}/{runs})")

                start = time.time()
                try:
                    result = runner(task_def)
                    elapsed = time.time() - start

                    # Extract metrics from result
                    resource_usage = result.get('resource_usage', {})
                    llm_metrics = resource_usage.get('llm_metrics', {})
                    comp_metrics = resource_usage.get('compensation_metrics', {})

                    run_result = {
                        "framework": framework,
                        "task": task_id,
                        "run": run_num + 1,
                        "llm_calls": llm_metrics.get('llm_call_count', 0),
                        "input_tokens": llm_metrics.get('total_input_tokens', 0),
                        "output_tokens": llm_metrics.get('total_output_tokens', 0),
                        "tokens": llm_metrics.get('total_tokens', 0),
                        "time": elapsed,
                        "rollback_triggered": comp_metrics.get('rollback_triggered', False),
                        "actions_compensated": comp_metrics.get('actions_compensated', 0),
                        "rollback_success": comp_metrics.get('compensation_success', False),
                        "status": "success"
                    }
                except Exception as e:
                    elapsed = time.time() - start
                    logger.error(f"Error running {framework} on {task_id}: {e}")
                    run_result = {
                        "framework": framework,
                        "task": task_id,
                        "run": run_num + 1,
                        "llm_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "tokens": 0,
                        "time": elapsed,
                        "rollback_triggered": False,
                        "actions_compensated": 0,
                        "rollback_success": False,
                        "status": "error",
                        "error": str(e)
                    }

                results["runs"].append(run_result)

    # Calculate summary statistics
    results["summary"] = calculate_summary(results["runs"])

    return results


def calculate_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from run results."""
    summary = {}

    # Group by framework and task
    for run in runs:
        key = f"{run['framework']}_{run['task']}"
        if key not in summary:
            summary[key] = {
                "framework": run["framework"],
                "task": run["task"],
                "runs": 0,
                "successes": 0,
                "total_llm_calls": 0,
                "total_tokens": 0,
                "total_time": 0,
                "rollback_successes": 0,
            }

        summary[key]["runs"] += 1
        if run["status"] == "success":
            summary[key]["successes"] += 1
        summary[key]["total_llm_calls"] += run["llm_calls"]
        summary[key]["total_tokens"] += run["tokens"]
        summary[key]["total_time"] += run["time"]
        if run["rollback_success"]:
            summary[key]["rollback_successes"] += 1

    # Calculate averages
    for key, stats in summary.items():
        if stats["runs"] > 0:
            stats["avg_llm_calls"] = stats["total_llm_calls"] / stats["runs"]
            stats["avg_tokens"] = stats["total_tokens"] / stats["runs"]
            stats["avg_time"] = stats["total_time"] / stats["runs"]
            stats["success_rate"] = stats["successes"] / stats["runs"]
            stats["rollback_success_rate"] = stats["rollback_successes"] / stats["runs"]

    return summary


def print_table(results: Dict[str, Any]):
    """Print results as a formatted table."""
    headers = ["Framework", "Task", "Run", "LLM Calls", "Tokens", "Time (s)", "Rollback"]
    rows = []

    for run in results["runs"]:
        rollback_status = "Success" if run["rollback_success"] else "Failed"
        if run["status"] == "error":
            rollback_status = "Error"

        rows.append([
            run["framework"],
            run["task"],
            run["run"],
            run["llm_calls"],
            run["tokens"],
            f"{run['time']:.2f}",
            rollback_status
        ])

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="pipe"))

    # Print summary
    if results.get("summary"):
        print("\n" + "=" * 80)
        print("SUMMARY (Averages)")
        print("=" * 80)
        summary_headers = ["Framework", "Task", "Avg LLM Calls", "Avg Tokens", "Avg Time (s)", "Rollback Rate"]
        summary_rows = []
        for key, stats in results["summary"].items():
            summary_rows.append([
                stats["framework"],
                stats["task"],
                f"{stats.get('avg_llm_calls', 0):.1f}",
                f"{stats.get('avg_tokens', 0):.0f}",
                f"{stats.get('avg_time', 0):.2f}",
                f"{stats.get('rollback_success_rate', 0)*100:.0f}%"
            ])
        print(tabulate(summary_rows, headers=summary_headers, tablefmt="pipe"))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark langchain-compensation vs SagaLLM on ACID transaction tasks"
    )
    parser.add_argument(
        "--frameworks",
        default="sagallm,compensation_lib,compensation_multiagent",
        help="Comma-separated list of frameworks to benchmark (default: all three)"
    )
    parser.add_argument(
        "--tasks",
        default="P5-ACID,P6-ACID,MONGODB-ACID",
        help="Comma-separated list of tasks to run (default: all ACID tasks)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per task per framework (default: 3)"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON file for detailed results (default: benchmark_results.json)"
    )

    args = parser.parse_args()

    # Parse arguments
    frameworks = [f.strip() for f in args.frameworks.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Validate environment
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not set. Gemini models will not work.")

    logger.info(f"Starting benchmark:")
    logger.info(f"  Frameworks: {frameworks}")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Runs per task: {args.runs}")
    logger.info(f"  Total runs: {len(frameworks) * len(tasks) * args.runs}")

    # Run benchmark
    results = run_benchmark(frameworks, tasks, args.runs)

    # Print results table
    print_table(results)

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDetailed results saved to: {args.output}")

    # Print LangSmith links
    print("\n" + "=" * 80)
    print("LANGSMITH TRACES")
    print("=" * 80)
    print("View traces at: https://smith.langchain.com")
    print("Projects:")
    print("  - realm-bench-sagallm")
    print("  - realm-bench-compensation-lib")
    print("  - realm-bench-compensation-multiagent")


if __name__ == "__main__":
    main()
