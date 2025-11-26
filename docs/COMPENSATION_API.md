# langchain-compensation API Reference

## Overview

The `langchain-compensation` library provides middleware for LangChain agents that automatically tracks tool calls and handles rollback on failure.

---

## Installation

```bash
# Requires Python 3.10+ (for langchain v1)
pip install langchain>=1.0.0 langgraph>=1.0.0 langchain-compensation
```

---

## Core API

### create_comp_agent

Factory function to create a LangChain agent with compensation middleware.

```python
from langchain_compensation import create_comp_agent

agent = create_comp_agent(
    model,                      # LangChain chat model
    tools,                      # List of tools
    compensation_mapping,       # Dict mapping tool -> compensation tool
    system_prompt=None,         # Optional system prompt
    state_mappers=None          # Optional state extraction functions
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `BaseChatModel` | LangChain chat model (e.g., ChatGoogleGenerativeAI) |
| `tools` | `List[Tool]` | List of LangChain tools |
| `compensation_mapping` | `Dict[str, str]` | Maps tool names to their compensation tool names |
| `system_prompt` | `str` (optional) | Custom system prompt for the agent |
| `state_mappers` | `Dict[str, Callable]` (optional) | Functions to extract compensation params from results |

**Returns:** A compiled LangGraph agent with compensation middleware.

**Example:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent

# Define tools
@tool
def book_flight(flight_id: str, passenger: str) -> str:
    """Book a flight."""
    return f"Booked flight {flight_id} for {passenger}"

@tool
def cancel_flight(booking_id: str) -> str:
    """Cancel a flight booking."""
    return f"Cancelled booking {booking_id}"

# Create agent
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

agent = create_comp_agent(
    model=llm,
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"}
)

# Use agent
result = agent.invoke({"messages": [HumanMessage(content="Book flight AA123")]})
```

---

### CompensationMiddleware

Middleware that intercepts tool calls and manages compensation.

```python
from langchain_compensation import CompensationMiddleware

middleware = CompensationMiddleware(
    compensation_mapping,       # Dict mapping tool -> compensation tool
    state_mappers=None          # Optional state extraction functions
)
```

**Methods:**

#### wrap_tool_call(tool, *args, **kwargs)

Wraps a tool call to track it in the compensation log.

```python
result = middleware.wrap_tool_call(book_flight, flight_id="AA123", passenger="John")
```

#### rollback()

Executes compensation for all tracked actions in dependency-aware order.

```python
middleware.rollback()  # Rolls back all tracked actions
```

#### get_log()

Returns the current compensation log.

```python
log = middleware.get_log()
for record in log.records:
    print(f"{record.tool_name}: {record.result}")
```

---

### CompensationLog

Thread-safe log of compensatable actions.

```python
from langchain_compensation import CompensationLog

log = CompensationLog()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `add(record)` | Add a CompensationRecord to the log |
| `get_records()` | Get all records |
| `clear()` | Clear all records |
| `get_rollback_order()` | Get records in DAG-based rollback order |

---

### CompensationRecord

Represents a single compensatable action.

```python
from langchain_compensation import CompensationRecord

record = CompensationRecord(
    tool_name="book_flight",
    params={"flight_id": "AA123", "passenger": "John"},
    result="booking_456",
    compensation_tool="cancel_flight",
    dependencies=[]  # Other records this depends on
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the executed tool |
| `params` | `Dict` | Parameters passed to the tool |
| `result` | `Any` | Result returned by the tool |
| `compensation_tool` | `str` | Name of the compensation tool |
| `dependencies` | `List[str]` | IDs of records this depends on |
| `timestamp` | `float` | Unix timestamp of execution |

---

## State Mappers

State mappers extract compensation parameters from tool results.

```python
def flight_state_mapper(result: str) -> Dict[str, Any]:
    """Extract booking ID from flight booking result."""
    # Parse "Booked flight AA123 for John, booking_id: BK456"
    booking_id = result.split("booking_id: ")[1]
    return {"booking_id": booking_id}

agent = create_comp_agent(
    model=llm,
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"},
    state_mappers={"book_flight": flight_state_mapper}
)
```

---

## Dependency Inference

The middleware automatically infers dependencies by analyzing data flow between tool calls.

**Example:**

```python
# Tool 1: book_flight returns "booking_123"
# Tool 2: reserve_car uses "booking_123" as parameter

# Middleware detects that Tool 2 depends on Tool 1's result
# On rollback: Tool 2's compensation runs before Tool 1's
```

**Rollback Order:**

```
Forward:  book_flight → reserve_car → charge_card → FAILURE
Rollback: refund_card → cancel_car → cancel_flight (respects dependencies)
```

---

## Error Handling

The middleware automatically triggers rollback when any tool raises an exception.

```python
@tool
def risky_operation(id: str) -> str:
    if some_condition:
        raise Exception("Operation failed!")
    return "success"

# If risky_operation fails, all previous compensatable actions
# are automatically rolled back
```

---

## Integration with LangSmith

Enable tracing for observability:

```python
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"
os.environ["LANGSMITH_API_KEY"] = "your-key"

# All tool calls and rollbacks are now traced
```

---

## Best Practices

1. **Define compensation for all state-changing tools**
   ```python
   COMPENSATION_MAPPING = {
       "book_flight": "cancel_flight",
       "reserve_hotel": "cancel_hotel",
       "charge_card": "refund_card",
       # Query tools don't need compensation
       # "get_weather": None  (not needed)
   }
   ```

2. **Use descriptive tool names**
   ```python
   # Good
   book_flight, cancel_flight_booking

   # Avoid
   tool1, undo_tool1
   ```

3. **Make compensation tools idempotent**
   ```python
   @tool
   def cancel_booking(booking_id: str) -> str:
       if not booking_exists(booking_id):
           return "Booking already cancelled"
       # Perform cancellation
   ```

4. **Handle partial failures in compensation**
   ```python
   @tool
   def refund_card(charge_id: str) -> str:
       try:
           return process_refund(charge_id)
       except RefundError:
           log_for_manual_review(charge_id)
           return "Refund queued for manual review"
   ```
