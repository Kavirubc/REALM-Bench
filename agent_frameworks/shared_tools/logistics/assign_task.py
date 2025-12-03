"""Action tool: Assign a task to an agent."""

from langchain_core.tools import tool
from typing import Optional, List

from ..state_manager import StateManager
from ..disruption_engine import DisruptionEngine
from ..logging_config import get_logger, log_tool_call, log_tool_result

logger = get_logger("realm_bench.tools")


@tool
def assign_task(
    task_id: str,
    assignee_id: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    dependencies: Optional[List[str]] = None
) -> str:
    """
    Assign a task to an agent or person.

    This is an ACTION tool that can be compensated by unassign_task.

    Args:
        task_id: Unique identifier for the task (e.g., "cook_turkey", "pickup_flowers")
        assignee_id: ID of the person/agent to assign (e.g., "mom", "guide_1")
        start_time: Start time for the task (optional)
        end_time: End time for the task (optional)
        dependencies: List of task IDs that must complete first (optional)

    Returns:
        Status message with assignment details
    """
    state = StateManager()
    disruption = DisruptionEngine()

    # Generate assignment ID
    assignment_id = f"task_{task_id}_{assignee_id}"

    log_tool_call(logger, "assign_task", {
        "task_id": task_id,
        "assignee_id": assignee_id
    })

    # Check for disruptions
    disruption_error = disruption.check_disruption("assign_task", {
        "task_id": task_id,
        "assignee_id": assignee_id
    })

    if disruption_error:
        log_tool_result(logger, "assign_task", False, disruption_error)
        return f"FAILED: {disruption_error}"

    # Create assignment data
    assignment_data = {
        "task_id": task_id,
        "assignee_id": assignee_id,
        "start_time": start_time,
        "end_time": end_time,
        "dependencies": dependencies or [],
        "status": "assigned"
    }

    # Try to create assignment
    success = state.assign_task(assignment_id, assignment_data)

    if success:
        state.record_action("assign_task", {
            "assignment_id": assignment_id,
            **assignment_data
        })

        time_str = ""
        if start_time is not None and end_time is not None:
            time_str = f" from {start_time} to {end_time}"
        elif start_time is not None:
            time_str = f" starting at {start_time}"

        dep_str = ""
        if dependencies:
            dep_str = f" (depends on: {', '.join(dependencies)})"

        result = (
            f"SUCCESS: Task {task_id} assigned to {assignee_id}"
            f"{time_str}{dep_str}. Assignment ID: {assignment_id}"
        )
        log_tool_result(logger, "assign_task", True, result)
        return result
    else:
        result = f"FAILED: Task assignment {assignment_id} already exists"
        log_tool_result(logger, "assign_task", False, result)
        return result
