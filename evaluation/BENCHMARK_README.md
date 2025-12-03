# Compensation Benchmark for LLM Agent Workflows

A comprehensive benchmark for evaluating compensation strategies in LLM agent workflows, designed for research publication.

## Overview

This benchmark extends REALM-Bench with systematic failure injection, controlled experiments, and enterprise workflow scenarios to compare tool-level compensation (langchain-compensation) vs agent-level compensation (SagaLLM).

## Features

- **Systematic Failure Injection**: Controlled failure injection with configurable rates and modes
- **Controlled Experiments**: Vary failure rates, workflow depth, parallelism
- **Enterprise Workflows**: Real-world scenarios (e-commerce, cloud provisioning, financial transactions)
- **Statistical Analysis**: Multiple runs with statistical summaries
- **Framework Comparison**: Head-to-head comparison of compensation strategies

## Quick Start

### Basic Usage

```bash
# Run benchmark on default tasks (P1, P5, P11) with default failure rates
python -m evaluation.compensation_benchmark

# Run specific tasks
python -m evaluation.compensation_benchmark --tasks P1 P5 E1 E2

# Run with custom failure rates
python -m evaluation.compensation_benchmark --failure-rates 0.0 0.1 0.25 0.5

# Run multiple iterations for statistical significance
python -m evaluation.compensation_benchmark --num-runs 20
```

### Advanced Usage

```python
from evaluation.compensation_benchmark import CompensationBenchmark, ExperimentConfig, FailureMode

benchmark = CompensationBenchmark(output_dir="results")

# Run a single experiment
config = ExperimentConfig(
    task_id="P5",
    framework="compensation_lib",
    failure_rate=0.25,
    failure_mode=FailureMode.TOOL_EXECUTION_ERROR,
    num_runs=10
)
summary = benchmark.run_experiment(config)

# Run full benchmark suite
results = benchmark.run_benchmark_suite(
    tasks=["P1", "P5", "P11", "E1", "E2", "E3"],
    frameworks=["compensation_lib", "sagallm"],
    failure_rates=[0.0, 0.1, 0.25, 0.5],
    num_runs=10
)
```

## Experiment Configuration

### Failure Modes

- `TOOL_EXECUTION_ERROR`: Tool returns error status
- `PARTIAL_SUCCESS`: Tool succeeds but with incomplete result
- `TIMEOUT`: Tool execution times out
- `CAPACITY_EXCEEDED`: Resource capacity exceeded
- `DEPENDENCY_FAILURE`: Dependent tool fails
- `CASCADING_FAILURE`: Multiple related failures

### Metrics Collected

- **Success Rate**: % of runs that completed successfully
- **Compensation Trigger Rate**: % of runs where compensation was triggered
- **Compensation Success Rate**: % of compensations that succeeded
- **Execution Time**: Average execution time (with std dev)
- **Compensation Time**: Time taken for rollback
- **Goal Satisfaction Rate**: % of goals achieved
- **Actions Compensated**: Average number of actions rolled back

## Enterprise Workflows

### E1: E-commerce Order Pipeline

- Reserve inventory → Process payment → Create shipment
- Tests cascading failures in transactional workflows

### E2: Cloud Infrastructure Provisioning

- Create VPC → Create subnet → Launch instance → Create security group
- Tests dependency ordering in infrastructure provisioning

### E3: Financial Transaction Processing

- Authenticate user → Transfer funds → Send notification
- Tests authentication and transaction rollback

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "timestamp": "20240101_120000",
  "config": {
    "tasks": ["P1", "P5", "P11"],
    "frameworks": ["compensation_lib", "sagallm"],
    "failure_rates": [0.0, 0.1, 0.25, 0.5],
    "num_runs": 10
  },
  "summaries": [
    {
      "config": {...},
      "num_runs": 10,
      "success_rate": 0.9,
      "avg_execution_time": 2.5,
      "compensation_trigger_rate": 0.3,
      "compensation_success_rate": 0.95,
      ...
    }
  ]
}
```

## Analysis

Use the comparison module to analyze results:

```python
from evaluation.compensation_benchmark import CompensationBenchmark

benchmark = CompensationBenchmark()
results = benchmark.run_benchmark_suite(...)
comparison = benchmark.compare_frameworks(results["summaries"])

print(comparison)
```

## Research Questions Addressed

1. **What classes of LLM tool invocations are safely compensatable?**

   - Tested across 11 planning tasks + 3 enterprise workflows

2. **What is the overhead of compensation tracking?**

   - Measured execution time with/without compensation

3. **How do different parameter extraction strategies affect recovery?**

   - Compare heuristic vs LLM-based extraction

4. **At what workflow complexity does tool-level outperform agent-level?**
   - Compare compensation_lib vs sagallm across task complexities

## Requirements

- Python 3.10+
- langchain-compensation
- REALM-Bench dependencies
- SagaLLM (optional, for comparison)

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{compensation_benchmark_2024,
  title={Compensating Actions in LLM Agent Workflows: A Benchmark Study},
  author={...},
  year={2024}
}
```
