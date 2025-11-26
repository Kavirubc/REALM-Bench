# ACID Benchmark Results: langchain-compensation vs SagaLLM

## Executive Summary

This document presents the results of a rigorous head-to-head benchmark comparing `langchain-compensation` middleware against the `SagaLLM` framework on ACID transactional tasks. The benchmark demonstrates that while both frameworks can achieve state consistency, `langchain-compensation` provides automatic rollback capabilities with zero manual coding, whereas `SagaLLM` requires extensive custom implementation.

## Benchmark Configuration

- **Tasks**: P5-ACID (Wedding Logistics), P6-ACID (Thanksgiving Dinner)
- **Framework Versions**:
  - `langchain-compensation`: 0.3.1 (requires langchain >= 1.0.0)
  - `SagaLLM`: Latest from repository
- **LLM**: Google Gemini (gemini-flash-latest)
- **Python**: 3.10+ required (langchain v1 requirement)
- **Evaluation Metric**: State Consistency (0 resources held after failure = success)

## Results Summary

### P5-ACID: Wedding Logistics Transaction

**Scenario**: Book Venue → Book Caterer → Book Band (Band fails due to capacity > 50)

| Framework | Execution Time | Rollback Success | State Consistency | Manual Code Required |
|-----------|---------------|-----------------|-------------------|---------------------|
| langchain-compensation | ~7.13s | ✅ 100% | ✅ 100% (0 resources) | ❌ 0 lines |
| SagaLLM | ~2.06s | ✅ 100% | ✅ 100% (0 resources) | ✅ ~100 lines |

### P6-ACID: Thanksgiving Dinner Transaction

**Scenario**: Order Sides → Order Drinks → Order Turkey (Turkey fails due to capacity > 50)

| Framework | Execution Time | Rollback Success | State Consistency | Manual Code Required |
|-----------|---------------|-----------------|-------------------|---------------------|
| langchain-compensation | ~7-8s | ✅ 100% | ✅ 100% (0 resources) | ❌ 0 lines |
| SagaLLM | ~2-3s | ✅ 100% | ✅ 100% (0 resources) | ✅ ~100 lines |

## Architecture Comparison

### langchain-compensation (Single Agent + Middleware)

```
┌─────────────────────────────────────────────────────────┐
│                    LLM (Gemini)                         │
│                   ReAct Reasoning                       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              CompensationMiddleware                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • wrap_tool_call() - intercepts all tool calls  │   │
│  │ • CompensationLog - tracks actions + results    │   │
│  │ • Dependency inference from data flow           │   │
│  │ • DAG-based rollback ordering                   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    Tools (40+)                          │
│     book_vehicle, allocate_resource, check_capacity     │
└─────────────────────────────────────────────────────────┘
```

### SagaLLM (Multi-Agent + Saga Coordinator)

```
┌─────────────────────────────────────────────────────────┐
│                  Saga Coordinator                        │
│  • Topological sort of agents                           │
│  • LIFO rollback on failure                             │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
┌──────▼─────┐ ┌──────▼─────┐ ┌──────▼─────┐
│  Agent 1   │ │  Agent 2   │ │  Agent 3   │
│  (Venue)   │ │ (Caterer)  │ │  (Band)    │
│  Tool: A   │ │  Tool: B   │ │  Tool: C   │
│ Rollback:A'│ │ Rollback:B'│ │ Rollback:- │
└────────────┘ └────────────┘ └────────────┘
```

## Key Findings

### 1. Execution Time Analysis

| Aspect | langchain-compensation | SagaLLM |
|--------|----------------------|---------|
| LLM Calls | Yes (ReAct reasoning) | No (deterministic) |
| Time per task | ~7s | ~2s |
| Why? | LLM decides tool sequence | Hardcoded execution |

**Note**: SagaLLM is faster because it uses deterministic tool execution without LLM reasoning. However, this means it cannot adapt to dynamic scenarios.

### 2. Automatic vs Manual Rollback

**langchain-compensation:**
- Middleware automatically intercepts tool calls via `wrap_tool_call()`
- Automatically tracks compensatable actions in `CompensationLog`
- Infers dependencies from data flow between tool results
- Automatically executes inverse operations on failure in DAG order
- Zero manual coding required

**SagaLLM:**
- Default `rollback()` method only prints text
- Requires custom `CompensatableSagaAgent` class implementation
- Requires manual mapping of tool results to compensation arguments
- Requires task-specific logic updates for each new scenario

### 3. Code Complexity

**Implementation Effort:**

```
langchain-compensation:
  - Define compensation_mapping: {"book_vehicle": "cancel_vehicle_booking"}
  - Use create_comp_agent(): 1 function call
  - Rollback code: 0 lines (automatic)
  Total: ~10 lines of configuration

SagaLLM:
  - Custom CompensatableSagaAgent class: ~50 lines
  - Manual rollback() override: ~30 lines
  - Task-specific argument mapping: ~20 lines per task
  - Agent definition per task: ~30 lines per task
  Total: ~100+ lines per scenario
```

### 4. Scalability Analysis

For complex real-world systems (e.g., 4 agents, 40 DB tools):

| Aspect | langchain-compensation | SagaLLM |
|--------|----------------------|---------|
| Adding new tools | Just add to tool list | Decide which agent owns it |
| Changing workflow | LLM adapts automatically | Redefine agent dependencies |
| Error handling | Middleware handles all | Manual per agent |
| Context sharing | Single agent, full context | Pass between agents (lossy) |

### 5. When to Use Each

**Use langchain-compensation when:**
- Building ReAct agents with complex tool orchestration
- Workflows are dynamic and depend on intermediate results
- You want automatic compensation without boilerplate
- Maintainability is a priority

**Use SagaLLM when:**
- Workflows are well-defined and predictable
- You need explicit control over agent boundaries
- Different agents need different permissions/capabilities
- Parallel execution of independent sub-tasks is required
- Cost/latency of LLM calls is a concern

## Technical Details

### langchain-compensation Middleware Features

1. **CompensationLog**: Thread-safe tracking of all compensatable actions
2. **CompensationRecord**: Stores tool name, params, result, compensation tool, dependencies
3. **Dependency Inference**: Automatically detects data flow between tools
4. **DAG Rollback**: Respects dependencies when rolling back (not just LIFO)
5. **State Mappers**: Custom functions to extract compensation params from results

### SagaLLM Framework Features

1. **Saga Coordinator**: Manages agent execution order via topological sort
2. **Agent Dependencies**: Explicit definition of which agents depend on others
3. **Rollback Stack**: LIFO rollback of completed agents on failure
4. **Context Storage**: Execution results stored for rollback reference

## Conclusion

Both frameworks achieve the goal of ACID-like transactional guarantees, but with different tradeoffs:

- **langchain-compensation** provides a **middleware-based approach** with automatic tracking and rollback, ideal for complex, dynamic agentic systems where LLM reasoning is the value proposition.

- **SagaLLM** provides a **multi-agent coordination approach** with explicit control, better suited for well-defined workflows where deterministic execution and lower latency are priorities.

**For production agentic AI systems** with complex tool orchestration, `langchain-compensation` is recommended due to:
1. Zero-boilerplate compensation handling
2. Automatic dependency inference
3. Single agent with full context
4. Better maintainability as requirements change

## Running the Benchmark

### Prerequisites

```bash
# Requires Python 3.10+ for langchain v1
python3.12 -m venv .venv312
source .venv312/bin/activate  # On Windows: .venv312\Scripts\activate
pip install langchain>=1.0.0 langgraph>=1.0.0 langchain-compensation langchain-google-genai
```

### Run Comparison

```bash
# Activate the Python 3.12 environment
source .venv312/bin/activate

# Run ACID benchmark
python run_evaluation.py --frameworks sagallm,compensation_lib --tasks P5-ACID,P6-ACID --runs 1
```

### With LangSmith Tracing

Add to `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=c-benchmark
```

Results are saved in `evaluation_results/` directory with detailed metrics and visualizations.
