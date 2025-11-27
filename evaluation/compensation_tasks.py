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
    ),

    "P5-ACID": TaskDefinition(
        task_id="P5-ACID",
        name="Wedding Logistics (ACID Transaction)",
        category=TaskCategory.LOGISTICS,
        description=(
            "Atomic Transaction: Book all wedding resources (Venue, Caterer, Band). "
            "The Band is unavailable. The system MUST cleanly rollback Venue and Caterer bookings "
            "to ensure state consistency. Partial success is considered a failure."
        ),
        goals=[
            TaskGoal("book_venue", "Book Wedding Venue", 0.3),
            TaskGoal("book_caterer", "Book Catering Service", 0.3),
            TaskGoal("book_band", "Book Live Band (Will Fail)", 0.4)
        ],
        constraints=[
            TaskConstraint(
                "atomic_transaction",
                "dependency",
                "All bookings must succeed or none should exist",
                {"dependencies": [
                    ["book_venue", "book_caterer"],
                    ["book_caterer", "book_band"]
                ]}
            )
        ],
        resources={
            "venue": {"id": "grand_hall", "capacity": 200},
            "caterer": {"id": "gourmet_delight", "menu": "standard"},
            "band": {"id": "the_rockers", "available": False} # Will cause failure
        },
        disruption_scenarios=[
            {"type": "resource_unavailable", "trigger": "book_band", "resource": "band"}
        ],
        evaluation_weights={
            "state_consistency": 1.0  # Only consistency matters for ACID test
        }
    ),

    "P6-ACID": TaskDefinition(
        task_id="P6-ACID",
        name="Thanksgiving Dinner (ACID Transaction)",
        category=TaskCategory.LOGISTICS,
        description=(
            "Atomic Transaction: Order Turkey, Sides, and Drinks. "
            "Turkey is out of stock. The system MUST cancel Sides and Drinks orders. "
            "No partial orders allowed."
        ),
        goals=[
            TaskGoal("order_sides", "Order Side Dishes", 0.3),
            TaskGoal("order_drinks", "Order Drinks", 0.3),
            TaskGoal("order_turkey", "Order Turkey (Will Fail)", 0.4)
        ],
        constraints=[
            TaskConstraint(
                "atomic_transaction",
                "dependency",
                "Main dish (Turkey) is required for any order",
                {"dependencies": [
                    ["order_sides", "order_turkey"], 
                    ["order_drinks", "order_turkey"]
                ]}
            )
        ],
        resources={
            "sides": ["mashed_potatoes", "stuffing"],
            "drinks": ["cider", "wine"],
            "turkey": {"available": False} # Will cause failure
        },
        disruption_scenarios=[
            {"type": "out_of_stock", "trigger": "order_turkey", "resource": "turkey"}
        ],
        evaluation_weights={
            "state_consistency": 1.0
        }
    ),
    
    "MONGODB-ACID": TaskDefinition(
        task_id="MONGODB-ACID",
        name="User Profile Creation (MongoDB ACID Transaction)",
        category=TaskCategory.LOGISTICS,
        description=(
            "Atomic Transaction: Create a complete user profile in MongoDB with multiple steps. "
            "1. Create user account"
            "2. Update user profile with additional information"
            "3. Add user preferences"
            "4. Create user session (will fail if user has 5+ active sessions)"
            "If session creation fails, all previous database operations must be rolled back "
            "to maintain database consistency. This tests real-world database transaction compensation."
        ),
        goals=[
            TaskGoal("create_user", "Create new user account in database", 0.25),
            TaskGoal("update_profile", "Update user profile information", 0.25),
            TaskGoal("add_preferences", "Add user preferences", 0.25),
            TaskGoal("create_session", "Create user session (Will Fail)", 0.25)
        ],
        constraints=[
            TaskConstraint(
                "atomic_transaction",
                "dependency",
                "All database operations must succeed or none should exist",
                {"dependencies": [
                    ["create_user", "update_profile"],
                    ["update_profile", "add_preferences"],
                    ["add_preferences", "create_session"]
                ]}
            ),
            TaskConstraint(
                "session_limit",
                "capacity",
                "User cannot have more than 5 active sessions",
                {"max_sessions": 5}
            )
        ],
        resources={
            "user_id": "test_user_123",
            "user_data": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "role": "premium"
            },
            "profile_updates": {
                "bio": "Software engineer and AI enthusiast",
                "location": "San Francisco, CA"
            },
            "preferences": {
                "theme": "dark",
                "notifications": True,
                "language": "en"
            },
            "session_data": {
                "device": "mobile",
                "ip": "192.168.1.100"
            },
            "existing_sessions": 5  # User already has 5 sessions, so new session will fail
        },
        disruption_scenarios=[
            {"type": "session_limit_exceeded", "trigger": "create_session", "max_sessions": 5}
        ],
        evaluation_weights={
            "state_consistency": 1.0  # Database consistency is critical
        }
    )
}


# Merge with main task definitions
def get_all_task_definitions() -> Dict[str, TaskDefinition]:
    """Get all task definitions including compensation tasks"""
    from .task_definitions import TASK_DEFINITIONS
    all_tasks = {**TASK_DEFINITIONS, **COMPENSATION_TASK_DEFINITIONS}
    return all_tasks
