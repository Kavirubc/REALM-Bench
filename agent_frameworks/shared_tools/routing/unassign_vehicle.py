"""Compensation tool: Unassign a vehicle."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..logging_config import get_logger, log_tool_call, log_tool_result, log_compensation

logger = get_logger("realm_bench.tools")


@tool
def unassign_vehicle(assignment_id: str) -> str:
    """
    Unassign a vehicle from its current assignment (compensation action).

    This is a COMPENSATION tool that reverses assign_vehicle.

    Args:
        assignment_id: Unique identifier for the assignment to cancel
            (format: "assign_{vehicle_id}_{passenger_id or route_id}")

    Returns:
        Status message indicating success or failure
    """
    state = StateManager()

    log_tool_call(logger, "unassign_vehicle", {"assignment_id": assignment_id})

    # Try to cancel the assignment
    cancelled = state.unassign_vehicle(assignment_id)

    if cancelled:
        state.record_compensation("unassign_vehicle", {
            "assignment_id": assignment_id,
            "cancelled_data": {
                "vehicle_id": cancelled.vehicle_id,
                "passenger_id": cancelled.passenger_id,
                "route_id": cancelled.route_id
            }
        }, original_action="assign_vehicle")

        log_compensation(
            logger,
            "assign_vehicle",
            "unassign_vehicle",
            f"Assignment {assignment_id} cancelled"
        )

        result = (
            f"SUCCESS: Assignment {assignment_id} cancelled. "
            f"Vehicle {cancelled.vehicle_id} is now available"
        )
        log_tool_result(logger, "unassign_vehicle", True, result)
        return result
    else:
        result = f"FAILED: Assignment {assignment_id} not found"
        log_tool_result(logger, "unassign_vehicle", False, result)
        return result
