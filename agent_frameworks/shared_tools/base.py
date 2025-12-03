"""
Base classes and data structures for shared tools.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ToolType(Enum):
    """Type of tool in the compensation pattern."""
    ACTION = "action"
    COMPENSATION = "compensation"
    QUERY = "query"


@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    success: bool
    tool_name: str
    tool_type: ToolType
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    compensatable: bool = True
    compensation_data: Dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        """Convert result to string format for agent consumption."""
        if self.success:
            return f"SUCCESS: {self.result_data}"
        else:
            return f"FAILED: {self.error_message}"


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""
    job_id: str
    machine_id: str
    start_time: int
    end_time: int
    duration: int
    priority: int = 1
    status: str = "scheduled"


@dataclass
class VehicleAssignment:
    """Represents a vehicle assignment."""
    assignment_id: str
    vehicle_id: str
    route_id: Optional[str] = None
    passenger_id: Optional[str] = None
    pickup_location: Optional[str] = None
    dropoff_location: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    status: str = "assigned"


@dataclass
class ResourceAllocation:
    """Represents a resource allocation."""
    allocation_id: str
    resource_type: str
    quantity: int
    destination: str
    allocated_by: Optional[str] = None
    status: str = "allocated"


@dataclass
class TeamDeployment:
    """Represents a team deployment."""
    deployment_id: str
    team_id: str
    region_id: str
    mission_type: str
    personnel_count: int = 0
    supplies: Dict[str, int] = field(default_factory=dict)
    status: str = "deployed"


@dataclass
class TaskAssignment:
    """Represents a task assignment."""
    assignment_id: str
    task_id: str
    assignee_id: str
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    dependencies: list = field(default_factory=list)
    status: str = "assigned"
