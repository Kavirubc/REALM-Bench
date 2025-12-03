"""
State manager for tracking tool execution state across the benchmark.

Provides a singleton pattern for consistent state access during task execution.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from threading import Lock
import copy
import time

from .base import (
    ScheduledJob,
    VehicleAssignment,
    ResourceAllocation,
    TeamDeployment,
    TaskAssignment,
)


@dataclass
class ExecutionState:
    """Current state of the planning execution."""
    scheduled_jobs: Dict[str, ScheduledJob] = field(default_factory=dict)
    assigned_vehicles: Dict[str, VehicleAssignment] = field(default_factory=dict)
    allocated_resources: Dict[str, ResourceAllocation] = field(default_factory=dict)
    deployed_teams: Dict[str, TeamDeployment] = field(default_factory=dict)
    task_assignments: Dict[str, TaskAssignment] = field(default_factory=dict)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    compensation_history: List[Dict[str, Any]] = field(default_factory=list)


class StateManager:
    """
    Thread-safe singleton state manager for tracking tool execution.

    Tracks all actions and compensations during task execution for
    metrics calculation and debugging.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._state = ExecutionState()
                    cls._instance._initialized = True
        return cls._instance

    def reset(self) -> None:
        """Reset state for new task execution."""
        with self._lock:
            self._state = ExecutionState()

    def get_state(self) -> ExecutionState:
        """Get a deep copy of current state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def record_action(self, action_name: str, action_data: Dict[str, Any]) -> None:
        """Record an action in the history."""
        with self._lock:
            self._state.action_history.append({
                "action": action_name,
                "data": action_data,
                "timestamp": time.time(),
                "type": "action"
            })

    def record_compensation(self, comp_name: str, comp_data: Dict[str, Any],
                           original_action: str) -> None:
        """Record a compensation in the history."""
        with self._lock:
            self._state.compensation_history.append({
                "compensation": comp_name,
                "data": comp_data,
                "original_action": original_action,
                "timestamp": time.time(),
                "type": "compensation"
            })

    # Scheduling operations
    def schedule_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """Schedule a job. Returns False if job already exists."""
        with self._lock:
            if job_id in self._state.scheduled_jobs:
                return False
            self._state.scheduled_jobs[job_id] = ScheduledJob(
                job_id=job_id,
                machine_id=job_data.get("machine_id", ""),
                start_time=job_data.get("start_time", 0),
                end_time=job_data.get("end_time", 0),
                duration=job_data.get("duration", 0),
                priority=job_data.get("priority", 1),
                status="scheduled"
            )
            return True

    def cancel_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Cancel a scheduled job. Returns the cancelled job or None."""
        with self._lock:
            return self._state.scheduled_jobs.pop(job_id, None)

    def get_scheduled_jobs(self) -> Dict[str, ScheduledJob]:
        """Get all scheduled jobs."""
        with self._lock:
            return copy.deepcopy(self._state.scheduled_jobs)

    # Routing operations
    def assign_vehicle(self, assignment_id: str,
                      assignment_data: Dict[str, Any]) -> bool:
        """Assign a vehicle. Returns False if assignment already exists."""
        with self._lock:
            if assignment_id in self._state.assigned_vehicles:
                return False
            self._state.assigned_vehicles[assignment_id] = VehicleAssignment(
                assignment_id=assignment_id,
                vehicle_id=assignment_data.get("vehicle_id", ""),
                route_id=assignment_data.get("route_id"),
                passenger_id=assignment_data.get("passenger_id"),
                pickup_location=assignment_data.get("pickup_location"),
                dropoff_location=assignment_data.get("dropoff_location"),
                start_time=assignment_data.get("start_time"),
                end_time=assignment_data.get("end_time"),
                status="assigned"
            )
            return True

    def unassign_vehicle(self, assignment_id: str) -> Optional[VehicleAssignment]:
        """Unassign a vehicle. Returns the cancelled assignment or None."""
        with self._lock:
            return self._state.assigned_vehicles.pop(assignment_id, None)

    def get_assigned_vehicles(self) -> Dict[str, VehicleAssignment]:
        """Get all vehicle assignments."""
        with self._lock:
            return copy.deepcopy(self._state.assigned_vehicles)

    # Resource operations
    def allocate_resource(self, allocation_id: str,
                         allocation_data: Dict[str, Any]) -> bool:
        """Allocate a resource. Returns False if allocation already exists."""
        with self._lock:
            if allocation_id in self._state.allocated_resources:
                return False
            self._state.allocated_resources[allocation_id] = ResourceAllocation(
                allocation_id=allocation_id,
                resource_type=allocation_data.get("resource_type", ""),
                quantity=allocation_data.get("quantity", 0),
                destination=allocation_data.get("destination", ""),
                allocated_by=allocation_data.get("allocated_by"),
                status="allocated"
            )
            return True

    def deallocate_resource(self, allocation_id: str) -> Optional[ResourceAllocation]:
        """Deallocate a resource. Returns the cancelled allocation or None."""
        with self._lock:
            return self._state.allocated_resources.pop(allocation_id, None)

    def get_allocated_resources(self) -> Dict[str, ResourceAllocation]:
        """Get all resource allocations."""
        with self._lock:
            return copy.deepcopy(self._state.allocated_resources)

    # Team deployment operations
    def deploy_team(self, deployment_id: str,
                   deployment_data: Dict[str, Any]) -> bool:
        """Deploy a team. Returns False if deployment already exists."""
        with self._lock:
            if deployment_id in self._state.deployed_teams:
                return False
            self._state.deployed_teams[deployment_id] = TeamDeployment(
                deployment_id=deployment_id,
                team_id=deployment_data.get("team_id", ""),
                region_id=deployment_data.get("region_id", ""),
                mission_type=deployment_data.get("mission_type", ""),
                personnel_count=deployment_data.get("personnel_count", 0),
                supplies=deployment_data.get("supplies", {}),
                status="deployed"
            )
            return True

    def recall_team(self, deployment_id: str) -> Optional[TeamDeployment]:
        """Recall a deployed team. Returns the cancelled deployment or None."""
        with self._lock:
            return self._state.deployed_teams.pop(deployment_id, None)

    def get_deployed_teams(self) -> Dict[str, TeamDeployment]:
        """Get all team deployments."""
        with self._lock:
            return copy.deepcopy(self._state.deployed_teams)

    # Task assignment operations
    def assign_task(self, assignment_id: str,
                   assignment_data: Dict[str, Any]) -> bool:
        """Assign a task. Returns False if assignment already exists."""
        with self._lock:
            if assignment_id in self._state.task_assignments:
                return False
            self._state.task_assignments[assignment_id] = TaskAssignment(
                assignment_id=assignment_id,
                task_id=assignment_data.get("task_id", ""),
                assignee_id=assignment_data.get("assignee_id", ""),
                start_time=assignment_data.get("start_time"),
                end_time=assignment_data.get("end_time"),
                dependencies=assignment_data.get("dependencies", []),
                status="assigned"
            )
            return True

    def unassign_task(self, assignment_id: str) -> Optional[TaskAssignment]:
        """Unassign a task. Returns the cancelled assignment or None."""
        with self._lock:
            return self._state.task_assignments.pop(assignment_id, None)

    def get_task_assignments(self) -> Dict[str, TaskAssignment]:
        """Get all task assignments."""
        with self._lock:
            return copy.deepcopy(self._state.task_assignments)

    # Utility methods
    def get_action_count(self) -> int:
        """Get total number of actions recorded."""
        with self._lock:
            return len(self._state.action_history)

    def get_compensation_count(self) -> int:
        """Get total number of compensations recorded."""
        with self._lock:
            return len(self._state.compensation_history)

    def get_all_history(self) -> List[Dict[str, Any]]:
        """Get combined action and compensation history sorted by timestamp."""
        with self._lock:
            combined = self._state.action_history + self._state.compensation_history
            return sorted(combined, key=lambda x: x.get("timestamp", 0))
