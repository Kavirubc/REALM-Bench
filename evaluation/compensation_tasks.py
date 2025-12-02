"""
Compensation Tasks for REALM-Bench

This module re-exports the core P1-P11 task definitions.
All tasks now use the planning_tools module for consistent tool-based execution.
"""

from typing import Dict
from .task_definitions import TaskDefinition, TASK_DEFINITIONS


def get_all_task_definitions() -> Dict[str, TaskDefinition]:
    """Get all P1-P11 task definitions."""
    return TASK_DEFINITIONS


# Re-export for backward compatibility
COMPENSATION_TASK_DEFINITIONS = {}  # Empty - no separate ACID tasks needed
