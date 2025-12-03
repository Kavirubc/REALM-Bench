"""Action tool: Allocate supplies to a region."""

from langchain_core.tools import tool
from typing import Optional

from ..state_manager import StateManager
from ..disruption_engine import DisruptionEngine
from ..logging_config import get_logger, log_tool_call, log_tool_result

logger = get_logger("realm_bench.tools")


@tool
def allocate_supplies(
    supply_type: str,
    quantity: int,
    destination_region: str,
    priority: str = "normal",
    transport_method: Optional[str] = None
) -> str:
    """
    Allocate supplies to a disaster region.

    This is an ACTION tool that can be compensated by return_supplies.

    Args:
        supply_type: Type of supplies (e.g., "medical", "food", "water", "shelter")
        quantity: Amount to allocate
        destination_region: Region receiving supplies (e.g., "region_1", "zone_A")
        priority: Priority level ("critical", "urgent", "normal")
        transport_method: Method of transport (e.g., "helicopter", "truck")

    Returns:
        Status message with allocation details
    """
    state = StateManager()
    disruption = DisruptionEngine()

    # Generate allocation ID
    allocation_id = f"supply_{supply_type}_{destination_region}_{quantity}"

    log_tool_call(logger, "allocate_supplies", {
        "supply_type": supply_type,
        "quantity": quantity,
        "destination_region": destination_region,
        "priority": priority
    })

    # Check for disruptions (uses allocate_resource disruption)
    disruption_error = disruption.check_disruption("allocate_resource", {
        "resource_type": supply_type,
        "destination": destination_region
    })

    if disruption_error:
        log_tool_result(logger, "allocate_supplies", False, disruption_error)
        return f"FAILED: {disruption_error}"

    # Create allocation data (reuse resource allocation)
    allocation_data = {
        "resource_type": supply_type,
        "quantity": quantity,
        "destination": destination_region,
        "allocated_by": f"disaster_relief_{priority}",
        "status": "allocated"
    }

    # Try to create allocation
    success = state.allocate_resource(allocation_id, allocation_data)

    if success:
        state.record_action("allocate_supplies", {
            "allocation_id": allocation_id,
            "supply_type": supply_type,
            "quantity": quantity,
            "destination_region": destination_region,
            "priority": priority,
            "transport_method": transport_method
        })

        transport_str = f" via {transport_method}" if transport_method else ""
        result = (
            f"SUCCESS: {quantity} units of {supply_type} allocated to "
            f"{destination_region} (priority: {priority}){transport_str}. "
            f"Allocation ID: {allocation_id}"
        )
        log_tool_result(logger, "allocate_supplies", True, result)
        return result
    else:
        result = f"FAILED: Supply allocation {allocation_id} already exists"
        log_tool_result(logger, "allocate_supplies", False, result)
        return result
