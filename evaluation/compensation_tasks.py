"""
Compensation-Specific Test Scenarios for REALM-Bench

This module defines test scenarios specifically designed to trigger and test
compensation/rollback capabilities in planning workflows.
"""

from typing import Dict, List, Any
from .task_definitions import (
    TaskDefinition, TaskGoal, TaskConstraint, TaskCategory, DisruptionType
)


# Compensation-specific task definitions
COMPENSATION_TASK_DEFINITIONS = {
    "CT1": TaskDefinition(
        task_id="CT1",
        name="Travel Booking with Payment Failure",
        category=TaskCategory.LOGISTICS,
        description=(
            "Book a flight and hotel for a trip. The payment processing step will fail, "
            "triggering automatic rollback of both bookings. This tests basic compensation "
            "functionality with a simple two-step workflow."
        ),
        goals=[
            TaskGoal("book_flight", "Book a flight to the destination", 0.4),
            TaskGoal("book_hotel", "Book a hotel at the destination", 0.3),
            TaskGoal("process_payment", "Process payment for bookings", 0.3)
        ],
        constraints=[
            TaskConstraint(
                "booking_deadline",
                "deadline",
                "All bookings must be completed within time limit",
                {"deadline": 60}
            ),
            TaskConstraint(
                "payment_required",
                "dependency",
                "Payment must be processed after bookings",
                {"dependencies": [["book_flight", "process_payment"], ["book_hotel", "process_payment"]]}
            )
        ],
        resources={
            "destinations": ["London", "Paris", "Tokyo"],
            "flight_options": [
                {"id": "flight_1", "destination": "London", "price": 500},
                {"id": "flight_2", "destination": "Paris", "price": 600}
            ],
            "hotel_options": [
                {"id": "hotel_1", "location": "London", "price": 200},
                {"id": "hotel_2", "location": "Paris", "price": 250}
            ],
            "payment_will_fail": True  # Flag to trigger payment failure
        },
        disruption_scenarios=[
            {"type": "payment_failure", "trigger": "process_payment", "amount": 1500}
        ],
        evaluation_weights={
            "planning_quality": 0.2,
            "planning_optimality": 0.2,
            "compensation_effectiveness": 0.6  # Focus on compensation
        }
    ),
    
    "CT2": TaskDefinition(
        task_id="CT2",
        name="Resource Allocation with Capacity Overflow",
        category=TaskCategory.SCHEDULING,
        description=(
            "Allocate multiple resources sequentially. The final allocation will exceed "
            "capacity, triggering rollback of all previous allocations. This tests "
            "compensation with dependency-aware ordering."
        ),
        goals=[
            TaskGoal("allocate_resource_1", "Allocate first resource", 0.25),
            TaskGoal("allocate_resource_2", "Allocate second resource", 0.25),
            TaskGoal("allocate_resource_3", "Allocate third resource", 0.25),
            TaskGoal("allocate_resource_4", "Allocate fourth resource (will fail)", 0.25)
        ],
        constraints=[
            TaskConstraint(
                "resource_capacity",
                "capacity",
                "Total resource allocation cannot exceed capacity",
                {"max_capacity": 50}
            ),
            TaskConstraint(
                "allocation_dependencies",
                "dependency",
                "Resources must be allocated in sequence",
                {"dependencies": [
                    ["allocate_resource_1", "allocate_resource_2"],
                    ["allocate_resource_2", "allocate_resource_3"],
                    ["allocate_resource_3", "allocate_resource_4"]
                ]}
            )
        ],
        resources={
            "resources": [
                {"id": "resource_1", "type": "compute", "amount": 10},
                {"id": "resource_2", "type": "storage", "amount": 15},
                {"id": "resource_3", "type": "network", "amount": 20},
                {"id": "resource_4", "type": "compute", "amount": 30}  # This will exceed capacity
            ],
            "total_capacity": 50
        },
        disruption_scenarios=[
            {"type": "capacity_overflow", "trigger": "allocate_resource_4", "requested": 30}
        ],
        evaluation_weights={
            "planning_quality": 0.2,
            "planning_optimality": 0.2,
            "compensation_effectiveness": 0.6
        }
    ),
    
    "CT3": TaskDefinition(
        task_id="CT3",
        name="Multi-Agent Coordination with Dependency Failures",
        category=TaskCategory.SCHEDULING,
        description=(
            "Coordinate multiple agents to complete tasks with dependencies. A later task "
            "will fail, requiring rollback of dependent tasks in correct dependency order. "
            "This tests DAG-based rollback ordering."
        ),
        goals=[
            TaskGoal("assign_task_1", "Assign task 1 to agent A", 0.2),
            TaskGoal("assign_task_2", "Assign task 2 to agent B (depends on task 1)", 0.2),
            TaskGoal("assign_task_3", "Assign task 3 to agent C (depends on task 2)", 0.2),
            TaskGoal("assign_task_4", "Assign task 4 to agent D (depends on task 3, will fail)", 0.2),
            TaskGoal("coordinate_agents", "Coordinate all agents successfully", 0.2)
        ],
        constraints=[
            TaskConstraint(
                "task_dependencies",
                "dependency",
                "Tasks must be assigned in dependency order",
                {"dependencies": [
                    ["assign_task_1", "assign_task_2"],
                    ["assign_task_2", "assign_task_3"],
                    ["assign_task_3", "assign_task_4"]
                ]}
            ),
            TaskConstraint(
                "agent_availability",
                "resource",
                "Each agent can only handle one task at a time",
                {"agents": ["agent_A", "agent_B", "agent_C", "agent_D"]}
            )
        ],
        resources={
            "agents": [
                {"id": "agent_A", "skills": ["task_1"], "available": True},
                {"id": "agent_B", "skills": ["task_2"], "available": True},
                {"id": "agent_C", "skills": ["task_3"], "available": True},
                {"id": "agent_D", "skills": ["task_4"], "available": False}  # Will cause failure
            ],
            "tasks": [
                {"id": "task_1", "duration": 10, "requires": []},
                {"id": "task_2", "duration": 15, "requires": ["task_1"]},
                {"id": "task_3", "duration": 20, "requires": ["task_2"]},
                {"id": "task_4", "duration": 25, "requires": ["task_3"]}
            ]
        },
        disruption_scenarios=[
            {"type": "agent_unavailable", "trigger": "assign_task_4", "agent": "agent_D"}
        ],
        evaluation_weights={
            "planning_quality": 0.2,
            "coordination_effectiveness": 0.3,
            "compensation_effectiveness": 0.5
        }
    )
}


# Merge with main task definitions
def get_all_task_definitions() -> Dict[str, TaskDefinition]:
    """Get all task definitions including compensation tasks"""
    from .task_definitions import TASK_DEFINITIONS
    all_tasks = {**TASK_DEFINITIONS, **COMPENSATION_TASK_DEFINITIONS}
    return all_tasks

