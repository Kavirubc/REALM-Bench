# LangSmith Tracing Setup for REALM-Bench

This document explains how LangSmith tracing is configured for the three frameworks being benchmarked.

## Framework Projects

Each framework has its own LangSmith project for easy filtering and analysis:

1. **SagaLLM**: `realm-bench-sagallm`
2. **Compensation Lib (Single-Agent)**: `realm-bench-compensation-lib`
3. **Compensation MultiAgent**: `realm-bench-compensation-multiagent`

## Viewing Traces in LangSmith

### 1. Access LangSmith
- Go to https://smith.langchain.com
- Make sure you're logged in and have `LANGSMITH_API_KEY` set in your environment

### 2. Filter by Project
In LangSmith, you can filter traces by project:
- Select the project dropdown
- Choose one of:
  - `realm-bench-sagallm`
  - `realm-bench-compensation-lib`
  - `realm-bench-compensation-multiagent`

### 3. Trace Structure

#### SagaLLM Traces
- **Top-level trace**: `sagallm-{task_id}` (e.g., `sagallm-MONGODB-ACID`)
  - Tags: `["sagallm", "multi-agent", "{task_id}", "mongodb-workflow"]`
  - Metadata: framework, task_id, task_name, agent_count, agent_names
- **Individual agent traces**: `sagallm-agent-{agent-name}` (e.g., `sagallm-agent-user-creation-agent`)
  - Tags: `["sagallm", "agent", "{agent-name}"]`
  - Metadata: agent_name, agent_type, has_compensation

#### Compensation Lib Traces
- **Single trace**: `compensation-lib-{task_id}` (e.g., `compensation-lib-MONGODB-ACID`)
  - Tags: `["compensation-lib", "single-agent", "{task_id}", "mongodb-workflow"]`
  - Metadata: framework, task_id, task_name, agent_type, workflow

#### Compensation MultiAgent Traces
- **Individual agent traces**: `compensation-multiagent-{task_id}-{agent-name}` (e.g., `compensation-multiagent-MONGODB-ACID-user-creation-agent`)
  - Tags: `["compensation-multiagent", "multi-agent", "{task_id}", "{agent-name}", "mongodb-workflow"]`
  - Metadata: framework, task_id, agent_name, agent_index, workflow

## Tags for Filtering

You can filter traces using these tags:
- `sagallm` - All SagaLLM traces
- `compensation-lib` - Single-agent compensation traces
- `compensation-multiagent` - Multi-agent compensation traces
- `mongodb-workflow` - All MongoDB workflow traces
- `single-agent` - Single-agent framework traces
- `multi-agent` - Multi-agent framework traces
- `{task_id}` - Specific task (e.g., `MONGODB-ACID`)

## Metadata Available

Each trace includes metadata for analysis:
- `framework`: Framework name
- `task_id`: Task identifier
- `task_name`: Human-readable task name
- `agent_name`: Agent name (for multi-agent)
- `agent_index`: Agent execution order (for multi-agent)
- `workflow`: Workflow type (e.g., "mongodb-user-profile")

## Running with Tracing

Tracing is automatically enabled when you run:
```bash
source .venv312/bin/activate
python run_evaluation.py --frameworks sagallm,compensation_lib,compensation_multiagent --tasks MONGODB-ACID --runs 1
```

Make sure `LANGSMITH_API_KEY` is set in your environment or `.env` file.

## Analyzing Traces

1. **Compare frameworks**: Filter by task_id tag to see all three frameworks side-by-side
2. **Agent-level analysis**: For multi-agent frameworks, look at individual agent traces
3. **Tool calls**: Expand traces to see individual tool calls and their results
4. **Error analysis**: Failed operations will show error messages in the trace
5. **Timing**: Each trace includes execution time for performance comparison

## Example Workflow

1. Run benchmark: `python run_evaluation.py --frameworks sagallm,compensation_lib,compensation_multiagent --tasks MONGODB-ACID --runs 1`
2. Open LangSmith: https://smith.langchain.com
3. Filter by project: Select `realm-bench-sagallm` (or other project)
4. Filter by tag: Add tag `MONGODB-ACID` to see all traces for this task
5. Compare: Switch between projects to compare framework behavior

