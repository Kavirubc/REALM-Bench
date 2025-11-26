# REALM-Bench Architecture Documentation

## Overview

REALM-Bench is a benchmarking framework for evaluating agentic AI systems with transactional guarantees. This document describes the architecture of the frameworks being compared.

---

## Framework Architectures

### 1. langchain-compensation (Single Agent + Middleware)

```
┌─────────────────────────────────────────────────────────┐
│                    LLM (Gemini)                         │
│                   ReAct Reasoning                       │
│                                                         │
│  "I need to book venue, then caterer, then check band"  │
└─────────────────────┬───────────────────────────────────┘
                      │ Tool calls
┌─────────────────────▼───────────────────────────────────┐
│              CompensationMiddleware                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ wrap_tool_call() - intercepts ALL tool calls    │   │
│  │                                                  │   │
│  │ For each tool call:                             │   │
│  │   1. Execute the tool                           │   │
│  │   2. Log to CompensationLog                     │   │
│  │   3. Infer dependencies from data flow          │   │
│  │   4. If failure → trigger DAG rollback          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  CompensationLog (Thread-safe)                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Record 1: book_vehicle(venue_1) → result_1      │   │
│  │ Record 2: allocate_resource(cat_1) → result_2   │   │
│  │ Record 3: check_capacity(band) → FAILED         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  On Failure → Rollback (respects dependencies):         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 1. deallocate_resource(cat_1)                   │   │
│  │ 2. cancel_vehicle_booking(venue_1)              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    Tools (40+)                          │
│     book_vehicle, allocate_resource, check_capacity     │
│     cancel_vehicle_booking, deallocate_resource         │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
- `create_comp_agent()`: Factory function that creates a LangChain agent with middleware
- `CompensationMiddleware`: Intercepts tool calls, tracks state, handles rollback
- `CompensationLog`: Thread-safe log of all compensatable actions
- `CompensationRecord`: Stores tool name, params, result, compensation tool, dependencies

---

### 2. SagaLLM (Multi-Agent + Saga Coordinator)

```
┌─────────────────────────────────────────────────────────┐
│                  Saga Coordinator                        │
│                                                         │
│  Responsibilities:                                      │
│  • Topologically sort agents based on dependencies      │
│  • Execute agents in order                              │
│  • On failure: LIFO rollback of completed agents        │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
       │ Sequential execution        │
       │              │              │
┌──────▼─────┐ ┌──────▼─────┐ ┌──────▼─────┐
│  Agent 1   │ │  Agent 2   │ │  Agent 3   │
│  (Venue)   │ │ (Caterer)  │ │  (Band)    │
│            │ │            │ │            │
│ Tool:      │ │ Tool:      │ │ Tool:      │
│ book_venue │ │ allocate   │ │ check_cap  │
│            │ │            │ │            │
│ Rollback:  │ │ Rollback:  │ │ Rollback:  │
│ cancel     │ │ deallocate │ │ (none)     │
└────────────┘ └────────────┘ └────────────┘

On Agent 3 Failure:
┌─────────────────────────────────────────────────────────┐
│  Rollback Stack (LIFO):                                 │
│  1. Agent 2.rollback() → deallocate_resource()          │
│  2. Agent 1.rollback() → cancel_vehicle_booking()       │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
- `SagaCoordinator`: Orchestrates agent execution and rollback
- `SagaAgent`: Individual agent with specific tool and rollback method
- `CompensatableSagaAgent`: Custom extension for actual compensation logic
- `ExecutionContext`: Stores results for rollback reference

---

## Comparison: Single Agent vs Multi-Agent

| Aspect | langchain-compensation | SagaLLM |
|--------|----------------------|---------|
| **Architecture** | Single agent + middleware | Multiple specialized agents |
| **Tool Assignment** | All tools available to one agent | Each agent owns specific tools |
| **Decision Making** | LLM decides tool sequence | Coordinator decides agent order |
| **Context** | Full context in single agent | Context passed between agents |
| **Rollback** | DAG-based (respects dependencies) | LIFO (last-in-first-out) |
| **Code Complexity** | ~10 lines configuration | ~100+ lines per scenario |

---

## Data Flow

### langchain-compensation Data Flow

```
User Request
    │
    ▼
┌─────────────┐
│    LLM      │ ──────────────────────────────────────┐
│  (ReAct)    │                                       │
└─────────────┘                                       │
    │                                                 │
    │ "Call book_vehicle(venue_1, wedding, main)"     │
    ▼                                                 │
┌─────────────────────────────────────┐              │
│     CompensationMiddleware          │              │
│                                     │              │
│  1. Execute book_vehicle()          │              │
│  2. Store in CompensationLog:       │              │
│     {tool: "book_vehicle",          │              │
│      params: {...},                 │              │
│      result: "booking_123",         │              │
│      compensation: "cancel_vehicle"}│              │
│  3. Return result to LLM            │              │
└─────────────────────────────────────┘              │
    │                                                 │
    │ result: "booking_123"                          │
    └─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────┐
│    LLM      │ "Now call allocate_resource..."
└─────────────┘
    │
    ... (continues until failure or completion)
```

### SagaLLM Data Flow

```
User Request
    │
    ▼
┌─────────────────────────────────────┐
│         Saga Coordinator            │
│                                     │
│  1. Parse agent dependencies        │
│  2. Topological sort                │
│  3. Execute in order                │
└─────────────────────────────────────┘
    │
    │ Execute Agent 1
    ▼
┌─────────────┐     Result      ┌─────────────┐
│   Agent 1   │ ───────────────►│  Context    │
│   (Venue)   │                 │  Storage    │
└─────────────┘                 └─────────────┘
    │
    │ Execute Agent 2
    ▼
┌─────────────┐     Result      ┌─────────────┐
│   Agent 2   │ ───────────────►│  Context    │
│  (Caterer)  │                 │  Storage    │
└─────────────┘                 └─────────────┘
    │
    │ Execute Agent 3
    ▼
┌─────────────┐
│   Agent 3   │ ───► FAILURE!
│   (Band)    │
└─────────────┘
    │
    │ Trigger rollback
    ▼
┌─────────────────────────────────────┐
│  Rollback (LIFO order):             │
│  Agent 2.rollback() using Context   │
│  Agent 1.rollback() using Context   │
└─────────────────────────────────────┘
```

---

## Scalability Considerations

### For Complex Systems (4 agents, 40+ tools)

**langchain-compensation Approach:**
```python
# Just add tools to the list
ALL_TOOLS = [tool1, tool2, ..., tool40]

# Define compensation mapping
COMPENSATION_MAPPING = {
    "book_flight": "cancel_flight",
    "reserve_hotel": "cancel_hotel",
    "charge_card": "refund_card",
    # ... etc
}

# Create agent - LLM handles orchestration
agent = create_comp_agent(
    model=llm,
    tools=ALL_TOOLS,
    compensation_mapping=COMPENSATION_MAPPING
)
```

**SagaLLM Approach:**
```python
# Must define each agent separately
agent1 = SagaAgent(name="FlightAgent", tools=[book_flight], ...)
agent2 = SagaAgent(name="HotelAgent", tools=[reserve_hotel], ...)
# ... 40+ agent definitions

# Must define explicit dependencies
agent2.depends_on(agent1)
agent3.depends_on(agent2)
# ... complex dependency graph

# Must implement custom rollback for each
class CustomAgent1(CompensatableSagaAgent):
    def rollback(self):
        # Manual implementation
        pass
```

---

## When to Use Each Architecture

### Use langchain-compensation when:
- Building ReAct agents with complex tool orchestration
- Workflows are dynamic and depend on intermediate results
- You want automatic compensation without boilerplate
- Single agent needs access to all tools
- Maintainability is a priority

### Use SagaLLM when:
- Workflows are well-defined and predictable
- You need explicit control over agent boundaries
- Different agents need different permissions/capabilities
- Parallel execution of independent sub-tasks is required
- Cost/latency of LLM calls is a concern
