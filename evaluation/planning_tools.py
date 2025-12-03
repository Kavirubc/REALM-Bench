"""
Planning Tools for REALM-Bench P1-P11 Tasks

This module provides compensatable tools for all 11 REALM-Bench planning scenarios.
Each tool has a corresponding compensation (rollback) function.

Supports failure injection for benchmark experiments.
"""

import json
import time
from typing import Dict, Any, Optional
from langchain_core.tools import tool

# Global failure injection state (set by benchmark)
_failure_injector = None
_failure_enabled = False

# ============================================================================
# State Tracking (simulates actual resource states)
# ============================================================================

_state = {
    "locations": {},
    "tours": {},
    "rides": {},
    "vehicles": {},
    "resources": {},
    "pickups": {},
    "relief_deployments": {},
    "supply_orders": {},
    "job_schedules": {},
}


def reset_state():
    """Reset all state for testing"""
    for key in _state:
        _state[key].clear()


# ============================================================================
# P1: Single-Agent Campus Tour - Routing
# ============================================================================

@tool
def visit_location(location_id: str, arrival_time: str) -> str:
    """Visit a campus location at a specific time. Returns visit confirmation."""
    # Check for failure injection (for benchmark experiments)
    if _failure_enabled and _failure_injector:
        if _failure_injector.should_fail("visit_location"):
            failure_result = _failure_injector.inject_failure("visit_location", None)
            return json.dumps(failure_result)
    
    visit_id = f"visit_{location_id}_{int(time.time())}"
    _state["locations"][visit_id] = {
        "location_id": location_id,
        "arrival_time": arrival_time,
        "status": "visited"
    }
    return json.dumps({"visit_id": visit_id, "location": location_id, "status": "visited"})


@tool
def cancel_visit(visit_id: str) -> str:
    """Cancel a scheduled visit to a location."""
    if visit_id in _state["locations"]:
        _state["locations"][visit_id]["status"] = "cancelled"
        return json.dumps({"visit_id": visit_id, "status": "cancelled"})
    return json.dumps({"visit_id": visit_id, "status": "not_found"})


@tool
def check_location_hours(location_id: str, requested_time: str) -> str:
    """Check if location is open. Fails if outside 9AM-5PM."""
    hour = int(requested_time.split(":")[0]) if ":" in requested_time else 12
    if hour < 9 or hour > 17:
        return json.dumps({"status": "error", "message": f"Location {location_id} closed at {requested_time}"})
    return json.dumps({"status": "available", "location": location_id})


# ============================================================================
# P2: Multi-Group Campus Tours - Scheduling
# ============================================================================

@tool
def assign_tour_guide(guide_id: str, group_id: str, start_time: str) -> str:
    """Assign a tour guide to a visitor group."""
    assignment_id = f"tour_{guide_id}_{group_id}_{int(time.time())}"
    _state["tours"][assignment_id] = {
        "guide_id": guide_id,
        "group_id": group_id,
        "start_time": start_time,
        "status": "assigned"
    }
    return json.dumps({"assignment_id": assignment_id, "guide": guide_id, "group": group_id, "status": "assigned"})


@tool
def cancel_tour_assignment(assignment_id: str) -> str:
    """Cancel a tour guide assignment."""
    if assignment_id in _state["tours"]:
        _state["tours"][assignment_id]["status"] = "cancelled"
        return json.dumps({"assignment_id": assignment_id, "status": "cancelled"})
    return json.dumps({"assignment_id": assignment_id, "status": "not_found"})


@tool
def check_guide_availability(guide_id: str, num_groups: int) -> str:
    """Check guide availability. Fails if > 2 groups assigned."""
    if num_groups > 2:
        return json.dumps({"status": "error", "message": f"Guide {guide_id} cannot handle {num_groups} groups"})
    return json.dumps({"status": "available", "guide": guide_id})


# ============================================================================
# P3: Urban Ride-Sharing - Routing
# ============================================================================

@tool
def book_ride(vehicle_id: str, passenger_id: str, pickup: str, dropoff: str) -> str:
    """Book a ride for a passenger."""
    booking_id = f"ride_{vehicle_id}_{passenger_id}_{int(time.time())}"
    _state["rides"][booking_id] = {
        "vehicle_id": vehicle_id,
        "passenger_id": passenger_id,
        "pickup": pickup,
        "dropoff": dropoff,
        "status": "booked"
    }
    return json.dumps({"booking_id": booking_id, "vehicle": vehicle_id, "status": "booked"})


@tool
def cancel_ride(booking_id: str) -> str:
    """Cancel a ride booking."""
    if booking_id in _state["rides"]:
        _state["rides"][booking_id]["status"] = "cancelled"
        return json.dumps({"booking_id": booking_id, "status": "cancelled"})
    return json.dumps({"booking_id": booking_id, "status": "not_found"})


@tool
def check_vehicle_capacity(vehicle_id: str, passengers_needed: int) -> str:
    """Check vehicle capacity. Fails if passengers > 4."""
    if passengers_needed > 4:
        return json.dumps({"status": "error", "message": f"Vehicle {vehicle_id} capacity exceeded"})
    return json.dumps({"status": "available", "vehicle": vehicle_id})


# ============================================================================
# P4: URS with Disruptions - Routing with dynamic changes
# ============================================================================

@tool
def book_ride_with_route(vehicle_id: str, passenger_id: str, route: str) -> str:
    """Book a ride with specific route."""
    booking_id = f"ride_{vehicle_id}_{int(time.time())}"
    _state["rides"][booking_id] = {
        "vehicle_id": vehicle_id,
        "passenger_id": passenger_id,
        "route": route,
        "status": "booked"
    }
    return json.dumps({"booking_id": booking_id, "route": route, "status": "booked"})


@tool
def update_route(booking_id: str, new_route: str) -> str:
    """Update route due to disruption. Fails for blocked routes."""
    if "blocked" in new_route.lower() or "closed" in new_route.lower():
        return json.dumps({"status": "error", "message": f"Route {new_route} is blocked"})
    if booking_id in _state["rides"]:
        _state["rides"][booking_id]["route"] = new_route
        return json.dumps({"booking_id": booking_id, "new_route": new_route, "status": "updated"})
    return json.dumps({"status": "error", "message": "Booking not found"})


# ============================================================================
# P5: Wedding Logistics
# ============================================================================

@tool
def book_venue(venue_id: str, event_type: str, guest_count: int) -> str:
    """Book a venue for an event."""
    booking_id = f"venue_{venue_id}_{int(time.time())}"
    _state["vehicles"][booking_id] = {
        "venue_id": venue_id,
        "event_type": event_type,
        "guest_count": guest_count,
        "status": "booked"
    }
    return json.dumps({"booking_id": booking_id, "venue": venue_id, "status": "booked"})


@tool
def cancel_venue(booking_id: str) -> str:
    """Cancel a venue booking."""
    if booking_id in _state["vehicles"]:
        _state["vehicles"][booking_id]["status"] = "cancelled"
        return json.dumps({"booking_id": booking_id, "status": "cancelled"})
    return json.dumps({"booking_id": booking_id, "status": "not_found"})


@tool
def book_catering(caterer_id: str, guest_count: int, menu_type: str) -> str:
    """Book catering service."""
    booking_id = f"catering_{caterer_id}_{int(time.time())}"
    _state["resources"][booking_id] = {
        "caterer_id": caterer_id,
        "guest_count": guest_count,
        "menu_type": menu_type,
        "status": "booked"
    }
    return json.dumps({"booking_id": booking_id, "caterer": caterer_id, "status": "booked"})


@tool
def cancel_catering(booking_id: str) -> str:
    """Cancel catering booking."""
    if booking_id in _state["resources"]:
        _state["resources"][booking_id]["status"] = "cancelled"
        return json.dumps({"booking_id": booking_id, "status": "cancelled"})
    return json.dumps({"booking_id": booking_id, "status": "not_found"})


@tool
def book_entertainment(band_id: str, event_duration: int, audience_size: int) -> str:
    """Book entertainment. Fails if audience > 50."""
    if audience_size > 50:
        return json.dumps({"status": "error", "message": f"Band {band_id} cannot accommodate {audience_size} guests"})
    booking_id = f"band_{band_id}_{int(time.time())}"
    _state["resources"][booking_id] = {"band_id": band_id, "status": "booked"}
    return json.dumps({"booking_id": booking_id, "band": band_id, "status": "booked"})


@tool
def cancel_entertainment(booking_id: str) -> str:
    """Cancel entertainment booking."""
    if booking_id in _state["resources"]:
        _state["resources"][booking_id]["status"] = "cancelled"
        return json.dumps({"booking_id": booking_id, "status": "cancelled"})
    return json.dumps({"booking_id": booking_id, "status": "not_found"})


# ============================================================================
# P6: Thanksgiving Dinner Planning
# ============================================================================

@tool
def schedule_airport_pickup(person_id: str, flight_time: str, driver_id: str) -> str:
    """Schedule airport pickup."""
    schedule_id = f"pickup_{person_id}_{int(time.time())}"
    _state["pickups"][schedule_id] = {
        "person_id": person_id,
        "flight_time": flight_time,
        "driver_id": driver_id,
        "status": "scheduled"
    }
    return json.dumps({"schedule_id": schedule_id, "person": person_id, "status": "scheduled"})


@tool
def cancel_pickup(schedule_id: str) -> str:
    """Cancel a pickup schedule."""
    if schedule_id in _state["pickups"]:
        _state["pickups"][schedule_id]["status"] = "cancelled"
        return json.dumps({"schedule_id": schedule_id, "status": "cancelled"})
    return json.dumps({"schedule_id": schedule_id, "status": "not_found"})


@tool
def assign_cooking_task(task_id: str, cook_id: str, start_time: str) -> str:
    """Assign a cooking task."""
    assignment_id = f"cook_{task_id}_{cook_id}_{int(time.time())}"
    _state["resources"][assignment_id] = {
        "task_id": task_id,
        "cook_id": cook_id,
        "start_time": start_time,
        "status": "assigned"
    }
    return json.dumps({"assignment_id": assignment_id, "task": task_id, "status": "assigned"})


@tool
def cancel_cooking_task(assignment_id: str) -> str:
    """Cancel a cooking task."""
    if assignment_id in _state["resources"]:
        _state["resources"][assignment_id]["status"] = "cancelled"
        return json.dumps({"assignment_id": assignment_id, "status": "cancelled"})
    return json.dumps({"assignment_id": assignment_id, "status": "not_found"})


@tool
def check_kitchen_capacity(cooks_needed: int) -> str:
    """Check kitchen capacity. Fails if > 3 cooks."""
    if cooks_needed > 3:
        return json.dumps({"status": "error", "message": "Kitchen can only accommodate 3 cooks"})
    return json.dumps({"status": "available", "capacity": 3 - cooks_needed})


# ============================================================================
# P7: Disaster Relief Deployment
# ============================================================================

@tool
def deploy_relief_team(team_id: str, region_id: str, resource_type: str) -> str:
    """Deploy a relief team to a region."""
    deployment_id = f"deploy_{team_id}_{region_id}_{int(time.time())}"
    _state["relief_deployments"][deployment_id] = {
        "team_id": team_id,
        "region_id": region_id,
        "resource_type": resource_type,
        "status": "deployed"
    }
    return json.dumps({"deployment_id": deployment_id, "team": team_id, "status": "deployed"})


@tool
def recall_relief_team(deployment_id: str) -> str:
    """Recall a deployed relief team."""
    if deployment_id in _state["relief_deployments"]:
        _state["relief_deployments"][deployment_id]["status"] = "recalled"
        return json.dumps({"deployment_id": deployment_id, "status": "recalled"})
    return json.dumps({"deployment_id": deployment_id, "status": "not_found"})


@tool
def allocate_relief_supplies(supply_type: str, amount: int, region_id: str) -> str:
    """Allocate supplies. Fails if amount > 1000."""
    if amount > 1000:
        return json.dumps({"status": "error", "message": f"Insufficient {supply_type}: only 1000 available"})
    alloc_id = f"supply_{supply_type}_{region_id}_{int(time.time())}"
    _state["resources"][alloc_id] = {"supply_type": supply_type, "amount": amount, "status": "allocated"}
    return json.dumps({"allocation_id": alloc_id, "supply": supply_type, "status": "allocated"})


@tool
def deallocate_relief_supplies(allocation_id: str) -> str:
    """Deallocate relief supplies."""
    if allocation_id in _state["resources"]:
        _state["resources"][allocation_id]["status"] = "deallocated"
        return json.dumps({"allocation_id": allocation_id, "status": "deallocated"})
    return json.dumps({"allocation_id": allocation_id, "status": "not_found"})


# ============================================================================
# P8: Wedding with Disruptions
# ============================================================================

@tool
def book_wedding_transport(vehicle_id: str, route: str, passenger_count: int) -> str:
    """Book wedding transport."""
    booking_id = f"transport_{vehicle_id}_{int(time.time())}"
    _state["vehicles"][booking_id] = {"vehicle_id": vehicle_id, "route": route, "status": "booked"}
    return json.dumps({"booking_id": booking_id, "vehicle": vehicle_id, "status": "booked"})


@tool
def cancel_wedding_transport(booking_id: str) -> str:
    """Cancel wedding transport."""
    if booking_id in _state["vehicles"]:
        _state["vehicles"][booking_id]["status"] = "cancelled"
        return json.dumps({"booking_id": booking_id, "status": "cancelled"})
    return json.dumps({"booking_id": booking_id, "status": "not_found"})


@tool
def check_route_status(route: str) -> str:
    """Check route. Fails for hotel-church route."""
    if "hotel-church" in route.lower():
        return json.dumps({"status": "error", "message": f"Route {route} is closed"})
    return json.dumps({"status": "available", "route": route})


# ============================================================================
# P9: Thanksgiving with Disruptions
# ============================================================================

@tool
def check_flight_status(flight_id: str) -> str:
    """Check flight status. flight1 is delayed 90min."""
    if "flight1" in flight_id.lower():
        return json.dumps({"status": "delayed", "flight": flight_id, "delay_minutes": 90})
    return json.dumps({"status": "on_time", "flight": flight_id})


@tool
def reschedule_pickup(schedule_id: str, new_time: str) -> str:
    """Reschedule pickup due to delay."""
    if schedule_id in _state["pickups"]:
        _state["pickups"][schedule_id]["flight_time"] = new_time
        return json.dumps({"schedule_id": schedule_id, "new_time": new_time, "status": "rescheduled"})
    return json.dumps({"status": "error", "message": "Schedule not found"})


# ============================================================================
# P10: GPU Supply Chain
# ============================================================================

@tool
def place_component_order(supplier_id: str, component: str, quantity: int) -> str:
    """Place component order."""
    order_id = f"order_{supplier_id}_{component}_{int(time.time())}"
    _state["supply_orders"][order_id] = {
        "supplier_id": supplier_id,
        "component": component,
        "quantity": quantity,
        "status": "ordered"
    }
    return json.dumps({"order_id": order_id, "supplier": supplier_id, "status": "ordered"})


@tool
def cancel_component_order(order_id: str) -> str:
    """Cancel component order."""
    if order_id in _state["supply_orders"]:
        _state["supply_orders"][order_id]["status"] = "cancelled"
        return json.dumps({"order_id": order_id, "status": "cancelled"})
    return json.dumps({"order_id": order_id, "status": "not_found"})


@tool
def check_supplier_capacity(supplier_id: str, quantity: int) -> str:
    """Check supplier capacity. Fails if quantity > 500."""
    if quantity > 500:
        return json.dumps({"status": "error", "message": f"Supplier {supplier_id} max capacity is 500"})
    return json.dumps({"status": "available", "supplier": supplier_id})


@tool
def schedule_assembly(facility_id: str, component: str, start_date: str) -> str:
    """Schedule assembly."""
    schedule_id = f"assembly_{facility_id}_{int(time.time())}"
    _state["job_schedules"][schedule_id] = {"facility_id": facility_id, "component": component, "status": "scheduled"}
    return json.dumps({"schedule_id": schedule_id, "facility": facility_id, "status": "scheduled"})


@tool
def cancel_assembly(schedule_id: str) -> str:
    """Cancel assembly."""
    if schedule_id in _state["job_schedules"]:
        _state["job_schedules"][schedule_id]["status"] = "cancelled"
        return json.dumps({"schedule_id": schedule_id, "status": "cancelled"})
    return json.dumps({"schedule_id": schedule_id, "status": "not_found"})


# ============================================================================
# P11: Job Shop Scheduling (JSSP)
# ============================================================================

@tool
def schedule_job(job_id: str, machine_id: str, start_time: int, duration: int) -> str:
    """Schedule a job on a machine."""
    schedule_id = f"job_{job_id}_{machine_id}_{int(time.time())}"
    _state["job_schedules"][schedule_id] = {
        "job_id": job_id,
        "machine_id": machine_id,
        "start_time": start_time,
        "duration": duration,
        "status": "scheduled"
    }
    return json.dumps({"schedule_id": schedule_id, "job": job_id, "machine": machine_id, "status": "scheduled"})


@tool
def cancel_job_schedule(schedule_id: str) -> str:
    """Cancel job schedule."""
    if schedule_id in _state["job_schedules"]:
        _state["job_schedules"][schedule_id]["status"] = "cancelled"
        return json.dumps({"schedule_id": schedule_id, "status": "cancelled"})
    return json.dumps({"schedule_id": schedule_id, "status": "not_found"})


@tool
def check_machine_availability(machine_id: str, start_time: int) -> str:
    """Check machine availability. machine_1 is down."""
    if machine_id == "machine_1" or machine_id == "1":
        return json.dumps({"status": "error", "message": f"Machine {machine_id} is down for maintenance"})
    return json.dumps({"status": "available", "machine": machine_id})


# ============================================================================
# Tool Collections and Compensation Mappings
# ============================================================================

P1_TOOLS = [visit_location, cancel_visit, check_location_hours]
P2_TOOLS = [assign_tour_guide, cancel_tour_assignment, check_guide_availability]
P3_TOOLS = [book_ride, cancel_ride, check_vehicle_capacity]
P4_TOOLS = [book_ride_with_route, cancel_ride, update_route]
P5_TOOLS = [book_venue, cancel_venue, book_catering, cancel_catering, book_entertainment, cancel_entertainment]
P6_TOOLS = [schedule_airport_pickup, cancel_pickup, assign_cooking_task, cancel_cooking_task, check_kitchen_capacity]
P7_TOOLS = [deploy_relief_team, recall_relief_team, allocate_relief_supplies, deallocate_relief_supplies]
P8_TOOLS = [book_wedding_transport, cancel_wedding_transport, check_route_status, book_venue, cancel_venue]
P9_TOOLS = [schedule_airport_pickup, cancel_pickup, check_flight_status, reschedule_pickup, assign_cooking_task, cancel_cooking_task]
P10_TOOLS = [place_component_order, cancel_component_order, check_supplier_capacity, schedule_assembly, cancel_assembly]
P11_TOOLS = [schedule_job, cancel_job_schedule, check_machine_availability]

ALL_TOOLS = list(set(
    P1_TOOLS + P2_TOOLS + P3_TOOLS + P4_TOOLS + P5_TOOLS +
    P6_TOOLS + P7_TOOLS + P8_TOOLS + P9_TOOLS + P10_TOOLS + P11_TOOLS
))

COMPENSATION_MAPPING = {
    "visit_location": "cancel_visit",
    "assign_tour_guide": "cancel_tour_assignment",
    "book_ride": "cancel_ride",
    "book_ride_with_route": "cancel_ride",
    "book_venue": "cancel_venue",
    "book_catering": "cancel_catering",
    "book_entertainment": "cancel_entertainment",
    "book_wedding_transport": "cancel_wedding_transport",
    "schedule_airport_pickup": "cancel_pickup",
    "assign_cooking_task": "cancel_cooking_task",
    "deploy_relief_team": "recall_relief_team",
    "allocate_relief_supplies": "deallocate_relief_supplies",
    "place_component_order": "cancel_component_order",
    "schedule_assembly": "cancel_assembly",
    "schedule_job": "cancel_job_schedule",
}

TASK_TOOLS = {
    "P1": P1_TOOLS,
    "P2": P2_TOOLS,
    "P3": P3_TOOLS,
    "P4": P4_TOOLS,
    "P5": P5_TOOLS,
    "P6": P6_TOOLS,
    "P7": P7_TOOLS,
    "P8": P8_TOOLS,
    "P9": P9_TOOLS,
    "P10": P10_TOOLS,
    "P11": P11_TOOLS,
}
