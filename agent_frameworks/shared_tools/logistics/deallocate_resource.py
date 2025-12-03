"""Compensation tool: Deallocate a resource."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..logging_config import get_logger, log_tool_call, log_tool_result, log_compensation

logger = get_logger("realm_bench.tools")


@tool
def deallocate_resource(allocation_id: str) -> str:
    """
    Deallocate a previously allocated resource (compensation action).

    This is a COMPENSATION tool that reverses allocate_resource.

    Args:
        allocation_id: Unique identifier for the allocation to cancel
            (format: "alloc_{resource_type}_{destination}_{quantity}")

    Returns:
        Status message indicating success or failure
    """
    state = StateManager()

    log_tool_call(logger, "deallocate_resource", {"allocation_id": allocation_id})

    # Try to cancel the allocation
    cancelled = state.deallocate_resource(allocation_id)

    if cancelled:
        state.record_compensation("deallocate_resource", {
            "allocation_id": allocation_id,
            "cancelled_data": {
                "resource_type": cancelled.resource_type,
                "quantity": cancelled.quantity,
                "destination": cancelled.destination
            }
        }, original_action="allocate_resource")

        log_compensation(
            logger,
            "allocate_resource",
            "deallocate_resource",
            f"Allocation {allocation_id} cancelled"
        )

        result = (
            f"SUCCESS: Deallocated {cancelled.quantity} units of "
            f"{cancelled.resource_type} from {cancelled.destination}. "
            f"Resources returned to pool"
        )
        log_tool_result(logger, "deallocate_resource", True, result)
        return result
    else:
        result = f"FAILED: Allocation {allocation_id} not found"
        log_tool_result(logger, "deallocate_resource", False, result)
        return result
