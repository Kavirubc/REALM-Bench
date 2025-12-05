"""Tool to unassign a task."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..logging_config import get_logger, log_tool_call, log_tool_result, log_compensation

logger = get_logger("realm_bench.tools")


@tool
def unassign_task(assignment_id: str) -> str:
    """
    Unassign a task from its current assignee, making the task available for reassignment.

    Args:
        assignment_id: Unique identifier for the task assignment
            (format: "task_{task_id}_{assignee_id}")

    Returns:
        Status message indicating success or failure
    """
    state = StateManager()

    log_tool_call(logger, "unassign_task", {"assignment_id": assignment_id})

    # Try to cancel the assignment
    cancelled = state.unassign_task(assignment_id)

    if cancelled:
        state.record_compensation("unassign_task", {
            "assignment_id": assignment_id,
            "cancelled_data": {
                "task_id": cancelled.task_id,
                "assignee_id": cancelled.assignee_id
            }
        }, original_action="assign_task")

        log_compensation(
            logger,
            "assign_task",
            "unassign_task",
            f"Task assignment {assignment_id} cancelled"
        )

        result = (
            f"SUCCESS: Task {cancelled.task_id} unassigned from "
            f"{cancelled.assignee_id}. Task is now available for reassignment"
        )
        log_tool_result(logger, "unassign_task", True, result)
        return result
    else:
        result = f"FAILED: Task assignment {assignment_id} not found"
        log_tool_result(logger, "unassign_task", False, result)
        return result
