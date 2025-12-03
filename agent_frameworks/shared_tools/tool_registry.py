"""
Tool registry for REALM-Bench.

Maps task categories to tools and provides compensation mappings for
langchain-compensation integration.
"""

import sys
import os
from typing import Dict, List, Callable, Any

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluation.task_definitions import TaskCategory

# Import all tools
from .scheduling import schedule_job, cancel_job
from .routing import assign_vehicle, unassign_vehicle
from .logistics import allocate_resource, deallocate_resource, assign_task, unassign_task
from .disaster_relief import deploy_team, recall_team, allocate_supplies, return_supplies


# Compensation mappings: action_tool_name -> compensation_tool_name
COMPENSATION_MAPPINGS: Dict[str, str] = {
    "schedule_job": "cancel_job",
    "assign_vehicle": "unassign_vehicle",
    "allocate_resource": "deallocate_resource",
    "assign_task": "unassign_task",
    "deploy_team": "recall_team",
    "allocate_supplies": "return_supplies",
}

# All available tools
ALL_TOOLS = [
    # Scheduling
    schedule_job,
    cancel_job,
    # Routing
    assign_vehicle,
    unassign_vehicle,
    # Logistics
    allocate_resource,
    deallocate_resource,
    assign_task,
    unassign_task,
    # Disaster Relief
    deploy_team,
    recall_team,
    allocate_supplies,
    return_supplies,
]

# Tools by category
CATEGORY_TOOLS: Dict[TaskCategory, List[Callable]] = {
    TaskCategory.SCHEDULING: [
        schedule_job,
        cancel_job,
    ],
    TaskCategory.ROUTING: [
        assign_vehicle,
        unassign_vehicle,
        schedule_job,  # For time-based scheduling
        cancel_job,
    ],
    TaskCategory.LOGISTICS: [
        allocate_resource,
        deallocate_resource,
        assign_task,
        unassign_task,
        assign_vehicle,
        unassign_vehicle,
    ],
    TaskCategory.DISASTER_RELIEF: [
        deploy_team,
        recall_team,
        allocate_supplies,
        return_supplies,
        allocate_resource,
        deallocate_resource,
    ],
    TaskCategory.SUPPLY_CHAIN: [
        allocate_resource,
        deallocate_resource,
        schedule_job,
        cancel_job,
        assign_task,
        unassign_task,
    ],
}


def get_tools_for_task(task_definition: Any) -> List[Callable]:
    """
    Get tools for a task based on its category.

    Args:
        task_definition: TaskDefinition with category field.

    Returns:
        List of tool functions appropriate for the task.
    """
    category = getattr(task_definition, 'category', None)
    if category and category in CATEGORY_TOOLS:
        return CATEGORY_TOOLS[category]
    # Default to all tools if category not found
    return ALL_TOOLS


def get_compensation_mapping_for_task(task_definition: Any) -> Dict[str, str]:
    """
    Get compensation mappings for tools used in a task.

    Args:
        task_definition: TaskDefinition with category field.

    Returns:
        Dictionary mapping action tool names to compensation tool names.
    """
    tools = get_tools_for_task(task_definition)
    tool_names = {getattr(t, 'name', str(t)) for t in tools}

    # Filter mappings to only include tools that are available for this task
    return {
        action: comp
        for action, comp in COMPENSATION_MAPPINGS.items()
        if action in tool_names
    }


def get_all_compensation_mappings() -> Dict[str, str]:
    """
    Get all compensation mappings.

    Returns:
        Complete dictionary of action -> compensation tool name mappings.
    """
    return COMPENSATION_MAPPINGS.copy()


def get_tool_by_name(name: str) -> Callable:
    """
    Get a tool function by its name.

    Args:
        name: Name of the tool.

    Returns:
        Tool function.

    Raises:
        ValueError: If tool not found.
    """
    for tool in ALL_TOOLS:
        tool_name = getattr(tool, 'name', None)
        if tool_name == name:
            return tool
    raise ValueError(f"Tool '{name}' not found")


def get_action_tools() -> List[Callable]:
    """Get all action tools (not compensation tools)."""
    action_names = set(COMPENSATION_MAPPINGS.keys())
    return [t for t in ALL_TOOLS if getattr(t, 'name', '') in action_names]


def get_compensation_tools() -> List[Callable]:
    """Get all compensation tools."""
    comp_names = set(COMPENSATION_MAPPINGS.values())
    return [t for t in ALL_TOOLS if getattr(t, 'name', '') in comp_names]
