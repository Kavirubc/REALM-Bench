"""Tool to allocate a resource."""

from langchain_core.tools import tool
from typing import Optional

from ..state_manager import StateManager
from ..disruption_engine import DisruptionEngine
from ..logging_config import get_logger, log_tool_call, log_tool_result

logger = get_logger("realm_bench.tools")


@tool
def allocate_resource(
    resource_type: str,
    quantity: int,
    destination: str,
    allocated_by: Optional[str] = None
) -> str:
    """
    Allocate a resource to a specific destination.

    Args:
        resource_type: Type of resource (e.g., "medical_supplies", "food", "water")
        quantity: Amount to allocate
        destination: Destination for the resource (e.g., "region_1", "warehouse_A")
        allocated_by: Entity allocating the resource (optional)

    Returns:
        Status message with allocation ID and details
    """
    state = StateManager()
    disruption = DisruptionEngine()

    # Generate allocation ID
    allocation_id = f"alloc_{resource_type}_{destination}_{quantity}"

    log_tool_call(logger, "allocate_resource", {
        "resource_type": resource_type,
        "quantity": quantity,
        "destination": destination
    })

    # Check for disruptions
    disruption_error = disruption.check_disruption("allocate_resource", {
        "resource_type": resource_type,
        "destination": destination
    })

    if disruption_error:
        log_tool_result(logger, "allocate_resource", False, disruption_error)
        return f"FAILED: {disruption_error}"

    # Create allocation data
    allocation_data = {
        "resource_type": resource_type,
        "quantity": quantity,
        "destination": destination,
        "allocated_by": allocated_by,
        "status": "allocated"
    }

    # Try to create allocation
    success = state.allocate_resource(allocation_id, allocation_data)

    if success:
        state.record_action("allocate_resource", {
            "allocation_id": allocation_id,
            **allocation_data
        })

        result = (
            f"SUCCESS: Allocated {quantity} units of {resource_type} "
            f"to {destination}. Allocation ID: {allocation_id}"
        )
        log_tool_result(logger, "allocate_resource", True, result)
        return result
    else:
        result = f"FAILED: Allocation {allocation_id} already exists"
        log_tool_result(logger, "allocate_resource", False, result)
        return result
