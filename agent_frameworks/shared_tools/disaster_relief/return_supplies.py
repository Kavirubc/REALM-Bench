"""Compensation tool: Return allocated supplies."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..logging_config import get_logger, log_tool_call, log_tool_result, log_compensation

logger = get_logger("realm_bench.tools")


@tool
def return_supplies(allocation_id: str) -> str:
    """
    Return previously allocated supplies (compensation action).

    This is a COMPENSATION tool that reverses allocate_supplies.

    Args:
        allocation_id: Unique identifier for the supply allocation
            (format: "supply_{supply_type}_{destination_region}_{quantity}")

    Returns:
        Status message indicating success or failure
    """
    state = StateManager()

    log_tool_call(logger, "return_supplies", {"allocation_id": allocation_id})

    # Try to cancel the allocation (uses resource deallocation)
    cancelled = state.deallocate_resource(allocation_id)

    if cancelled:
        state.record_compensation("return_supplies", {
            "allocation_id": allocation_id,
            "cancelled_data": {
                "supply_type": cancelled.resource_type,
                "quantity": cancelled.quantity,
                "destination": cancelled.destination
            }
        }, original_action="allocate_supplies")

        log_compensation(
            logger,
            "allocate_supplies",
            "return_supplies",
            f"Supply allocation {allocation_id} cancelled"
        )

        result = (
            f"SUCCESS: {cancelled.quantity} units of {cancelled.resource_type} "
            f"returned from {cancelled.destination}. Supplies available for reallocation"
        )
        log_tool_result(logger, "return_supplies", True, result)
        return result
    else:
        result = f"FAILED: Supply allocation {allocation_id} not found"
        log_tool_result(logger, "return_supplies", False, result)
        return result
