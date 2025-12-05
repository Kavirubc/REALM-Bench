"""Tool to assign a vehicle to a route or passenger."""

from langchain_core.tools import tool
from typing import Optional

from ..state_manager import StateManager
from ..disruption_engine import DisruptionEngine
from ..logging_config import get_logger, log_tool_call, log_tool_result

logger = get_logger("realm_bench.tools")


@tool
def assign_vehicle(
    vehicle_id: str,
    passenger_id: Optional[str] = None,
    route_id: Optional[str] = None,
    pickup_location: Optional[str] = None,
    dropoff_location: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> str:
    """
    Assign a vehicle to a passenger or route for transportation.

    Args:
        vehicle_id: Unique identifier for the vehicle (e.g., "vehicle_1", "V1")
        passenger_id: ID of passenger to pick up (optional)
        route_id: ID of route to assign (optional)
        pickup_location: Pickup location name (optional)
        dropoff_location: Dropoff location name (optional)
        start_time: Start time of assignment (optional)
        end_time: End time of assignment (optional)

    Returns:
        Status message indicating success or failure with assignment details
    """
    state = StateManager()
    disruption = DisruptionEngine()

    # Generate assignment ID
    assignment_id = f"assign_{vehicle_id}_{passenger_id or route_id or 'general'}"

    log_tool_call(logger, "assign_vehicle", {
        "vehicle_id": vehicle_id,
        "passenger_id": passenger_id,
        "route_id": route_id
    })

    # Check for disruptions
    disruption_error = disruption.check_disruption("assign_vehicle", {
        "vehicle_id": vehicle_id,
        "route_id": route_id
    })

    if disruption_error:
        log_tool_result(logger, "assign_vehicle", False, disruption_error)
        return f"FAILED: {disruption_error}"

    # Create assignment data
    assignment_data = {
        "vehicle_id": vehicle_id,
        "passenger_id": passenger_id,
        "route_id": route_id,
        "pickup_location": pickup_location,
        "dropoff_location": dropoff_location,
        "start_time": start_time,
        "end_time": end_time,
        "status": "assigned"
    }

    # Try to create assignment
    success = state.assign_vehicle(assignment_id, assignment_data)

    if success:
        state.record_action("assign_vehicle", {
            "assignment_id": assignment_id,
            **assignment_data
        })

        details = []
        if passenger_id:
            details.append(f"passenger {passenger_id}")
        if route_id:
            details.append(f"route {route_id}")
        if pickup_location and dropoff_location:
            details.append(f"from {pickup_location} to {dropoff_location}")

        result = (
            f"SUCCESS: Vehicle {vehicle_id} assigned to "
            f"{', '.join(details) if details else 'task'}. "
            f"Assignment ID: {assignment_id}"
        )
        log_tool_result(logger, "assign_vehicle", True, result)
        return result
    else:
        result = f"FAILED: Assignment {assignment_id} already exists"
        log_tool_result(logger, "assign_vehicle", False, result)
        return result
