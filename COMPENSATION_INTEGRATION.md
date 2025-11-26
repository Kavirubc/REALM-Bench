# REALM-Bench Integration with LangChain Compensation

## 1. Introduction

This document provides a comprehensive technical overview of the integration between the `langchain-compensation` library and the `REALM-Bench` framework. The primary objective of this integration is to evaluate the reliability and effectiveness of automatic rollback mechanisms in agentic workflows.

The system implements the Saga pattern for distributed transactions within an agentic context, allowing for:
1.  **Automatic Failure Detection**: Identifying when a tool execution fails.
2.  **State Tracking**: maintaining a log of all compensatable actions.
3.  **Automatic Rollback**: Reversing completed actions in Last-In-First-Out (LIFO) or Dependency-Aware Graph (DAG) order when a subsequent step fails.

---

## 2. Architecture Overview

The integration introduces a dedicated framework runner that operates alongside existing runners (LangGraph, AutoGen, etc.). It embeds the compensation middleware logic directly into the tool execution flow.

### System Components

```mermaid
classDiagram
    class BaseFrameworkRunner {
        <<Abstract>>
        +_create_execution_result()
        +_record_memory_usage()
    }

    class CompensationLangGraphRunner {
        +CompensationLog compensation_log
        +List[Tool] tools
        +Dict[str, str] compensation_mapping
        +call(task_definition)
        -_rollback_on_error()
        -_execute_compensation()
    }

    class CompensationLog {
        +add(record)
        +update(record_id, status)
        +get_rollback_plan()
        +mark_compensated(record_id)
    }

    class CompensationMiddleware {
        <<Logic>>
        +Interceds Tool Calls
        +Tracks Dependencies
        +Triggers Rollback
    }

    class CompensationMetrics {
        +evaluate_rollback_success_rate()
        +evaluate_compensation_coverage()
        +evaluate_rollback_time()
    }

    BaseFrameworkRunner <|-- CompensationLangGraphRunner
    CompensationLangGraphRunner *-- CompensationLog
    CompensationLangGraphRunner ..> CompensationMiddleware : implements logic of
    CompensationLangGraphRunner ..> CompensationMetrics : generates data for
```

### Execution Flow (Sequence Diagram)

The following diagram illustrates the lifecycle of a task that encounters a failure, triggering the compensation mechanism.

```mermaid
sequenceDiagram
    participant Runner as CompensationRunner
    participant Agent as LLM Agent
    participant Log as CompensationLog
    participant ToolA as Tool: Book Resource
    participant ToolB as Tool: Process Payment
    participant CompTool as Tool: Cancel Resource

    Runner->>Agent: Invoke with Task (CT1)
    Agent->>ToolA: Call book_resource()
    activate ToolA
    ToolA-->>Runner: Execute Tool A
    Runner->>Log: Add Record (PENDING)
    Runner->>ToolA: Invoke Logic
    ToolA-->>Runner: Success (Result)
    Runner->>Log: Update Record (COMPLETED)
    Runner-->>Agent: Tool A Result
    deactivate ToolA

    Agent->>ToolB: Call process_payment()
    activate ToolB
    ToolB-->>Runner: Execute Tool B
    Runner->>ToolB: Invoke Logic
    ToolB-->>Runner: Error (Payment Failed)
    Runner->>Log: Log Failure
    
    rect rgb(255, 230, 230)
        Note over Runner, Log: Error Detected - Triggering Rollback
        Runner->>Log: get_rollback_plan()
        Log-->>Runner: [Record A]
        
        Runner->>CompTool: Invoke cancel_resource(params from A)
        activate CompTool
        CompTool-->>Runner: Success
        deactivate CompTool
        
        Runner->>Log: mark_compensated(Record A)
    end
    
    Runner-->>Agent: Return Failure & Compensation Status
    deactivate ToolB
```

---

## 3. Core Implementation Details

### 3.1 Compensation Framework Runner
**File:** `evaluation/compensation_runner.py`

The `CompensationLangGraphRunner` is the core engine of this integration. Unlike standard runners that simply execute a graph, this runner wraps the tool execution node to inject middleware logic.

**Key Responsibilities:**
1.  **Tool Binding**: It binds a set of 10 specialized tools (e.g., `book_vehicle`, `cancel_vehicle_booking`) to the Large Language Model (LLM).
2.  **Interception**: It uses a custom `compensating_tool_node` function instead of the standard LangGraph `ToolNode`. This function intercepts every tool call before and after execution.
3.  **Logging**: Before execution, it creates a `CompensationRecord` in the `CompensationLog` if the tool is marked as compensatable.
4.  **Error Handling**: If a tool returns an error status or raises an exception, the runner immediately triggers `_rollback_on_error()`.

### 3.2 Compensation Middleware Logic
The logic originally contained in the `langchain-compensation` library was embedded directly into the runner to ensure compatibility and fine-grained control over the evaluation process.

**Rollback Strategy:**
*   **LIFO (Last-In-First-Out)**: The default strategy is to reverse actions in the reverse order of their completion.
*   **State Tracking**: The system tracks `_vehicle_bookings`, `_resource_allocations`, etc., in global dictionaries to simulate a persistent database state. This allows verification that rollbacks actually reverted the system state.

### 3.3 Compensation Metrics
**File:** `evaluation/compensation_metrics.py`

We introduced a new class of metrics specifically for transactional integrity:

*   **Rollback Success Rate**: The percentage of failed workflows where the system successfully reverted to a consistent state.
    *   Formula: `(Success Count / Total Attempted Compensations) * 100`
*   **Compensation Coverage**: The percentage of executed actions that had a defined compensation handler. This measures how "safe" the workflow is.
*   **Compensation Efficiency**: A measure of the system's speed in handling rollbacks, calculated as compensations performed per second.

---

## 4. Test Scenarios

To rigorously test the system, we defined three new scenarios in `evaluation/compensation_tasks.py`.

### CT1: Travel Booking with Payment Failure
*   **Workflow**: Book Flight -> Book Hotel -> Process Payment.
*   **Failure Trigger**: The `process_payment` tool is hardcoded to fail for specific inputs.
*   **Expected Behavior**: The system must detect the payment failure and automatically invoke `cancel_hotel` and `cancel_flight`.

### CT2: Resource Allocation with Capacity Overflow
*   **Workflow**: Allocate Resource 1 -> 2 -> 3 -> 4.
*   **Failure Trigger**: Resource 4 requests an amount that exceeds the total global capacity (`check_capacity` tool).
*   **Expected Behavior**: All previous allocations (1, 2, 3) must be deallocated to free up the reserved capacity.

### CT3: Multi-Agent Coordination
*   **Workflow**: Assign Task 1 -> 2 -> 3 -> 4 to different agents.
*   **Failure Trigger**: Agent 4 is marked as "unavailable" in the resource definition.
*   **Expected Behavior**: The system must unassign tasks from Agents 3, 2, and 1.

---

## 5. Usage Guide

### Running the Evaluation
To run the benchmark specifically for the compensation framework:

```bash
python run_evaluation.py --frameworks compensation --tasks CT1,CT2,CT3
```

### Interpreting Results
The output JSON files in `evaluation_results/` now contain a `compensation_metrics` object within the `resource_usage` block:

```json
"compensation_metrics": {
    "rollback_count": 1,
    "compensation_success_count": 2,
    "compensation_failure_count": 0,
    "compensation_log_size": 2
}
```

This indicates that 1 failure triggered a rollback process, which successfully compensated 2 distinct actions (e.g., cancelling a flight and a hotel).

---

## 6. Conclusion

This integration successfully demonstrates that LLM-based agents can be made reliable through middleware that enforces transactional guarantees. The `CompensationLangGraphRunner` provides a robust harness for testing these capabilities against complex, multi-step failure scenarios.

