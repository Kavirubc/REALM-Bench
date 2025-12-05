"""
Disruption engine for injecting failures during tool execution.

Maps task disruption scenarios to tool failures for testing compensation behavior.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import logging

from .logging_config import get_logger, log_disruption

logger = get_logger("realm_bench.disruption")


class DisruptionTrigger(Enum):
    """When a disruption should trigger."""
    ON_TOOL_CALL = "on_tool_call"
    AFTER_N_ACTIONS = "after_n_actions"
    PROBABILISTIC = "probabilistic"


@dataclass
class DisruptionConfig:
    """Configuration for a single disruption."""
    disruption_type: str
    affected_tool: Optional[str] = None
    affected_resource: Optional[str] = None
    trigger: DisruptionTrigger = DisruptionTrigger.ON_TOOL_CALL
    trigger_after_n_actions: int = 1
    probability: float = 1.0
    error_message: str = "Disruption occurred"
    triggered: bool = False


# Mapping of disruption types to affected tools
DISRUPTION_TOOL_MAPPING = {
    "machine_breakdown": {
        "tool": "schedule_job",
        "message": "Machine unavailable due to breakdown"
    },
    "traffic_delay": {
        "tool": "assign_vehicle",
        "message": "Route blocked due to traffic delay"
    },
    "road_closure": {
        "tool": "assign_vehicle",
        "message": "Road closed, route unavailable"
    },
    "flight_delay": {
        "tool": "allocate_resource",
        "message": "Resource delayed due to flight delay"
    },
    "resource_shortage": {
        "tool": "allocate_resource",
        "message": "Resource unavailable due to shortage"
    },
    "weather_event": {
        "tool": "deploy_team",
        "message": "Deployment blocked due to weather conditions"
    },
}


class DisruptionEngine:
    """
    Manages disruption injection during tool execution.

    Loads disruption scenarios from task definitions and triggers them
    during tool calls to test compensation behavior.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._disruptions = []
            cls._instance._action_count = 0
            cls._instance._triggered_disruptions = []
            cls._instance._enabled = True
        return cls._instance

    def reset(self) -> None:
        """Reset the disruption engine for a new task."""
        self._disruptions = []
        self._action_count = 0
        self._triggered_disruptions = []

    def enable(self) -> None:
        """Enable disruption injection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable disruption injection."""
        self._enabled = False

    def configure_from_task(self, task_definition) -> None:
        """
        Load disruption scenarios from a task definition.

        Args:
            task_definition: TaskDefinition with disruption_scenarios field.
        """
        self._disruptions = []

        if not hasattr(task_definition, "disruption_scenarios"):
            return

        for scenario in task_definition.disruption_scenarios:
            config = self._map_scenario_to_config(scenario)
            if config:
                self._disruptions.append(config)
                logger.info(
                    f"Configured disruption: {config.disruption_type} "
                    f"-> {config.affected_tool}"
                )

    def _map_scenario_to_config(
        self,
        scenario: Dict[str, Any]
    ) -> Optional[DisruptionConfig]:
        """Map a task disruption scenario to engine config."""
        dtype = scenario.get("type")

        # Handle enum types
        if hasattr(dtype, "value"):
            dtype = dtype.value

        dtype_lower = str(dtype).lower() if dtype else ""

        # Look up in mapping
        if dtype_lower in DISRUPTION_TOOL_MAPPING:
            mapping = DISRUPTION_TOOL_MAPPING[dtype_lower]
            return DisruptionConfig(
                disruption_type=dtype_lower,
                affected_tool=mapping["tool"],
                error_message=f"{mapping['message']}: {scenario}",
                trigger=DisruptionTrigger.AFTER_N_ACTIONS,
                trigger_after_n_actions=2,  # Trigger after 2nd action
                probability=1.0  # 100% chance to trigger
            )

        return None

    def check_disruption(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check if a disruption should occur for this tool call.

        Args:
            tool_name: Name of the tool being called.
            tool_args: Arguments passed to the tool.

        Returns:
            Error message if disruption triggers, None otherwise.
        """
        if not self._enabled:
            return None

        self._action_count += 1

        for disruption in self._disruptions:
            if disruption.triggered:
                continue

            if self._should_trigger(disruption, tool_name):
                disruption.triggered = True
                self._triggered_disruptions.append({
                    "disruption": disruption.disruption_type,
                    "tool": tool_name,
                    "args": tool_args,
                    "action_count": self._action_count,
                    "error_message": disruption.error_message
                })
                log_disruption(
                    logger,
                    disruption.disruption_type,
                    tool_name,
                    disruption.error_message
                )
                return disruption.error_message

        return None

    def _should_trigger(
        self,
        disruption: DisruptionConfig,
        tool_name: str
    ) -> bool:
        """Determine if a disruption should trigger."""
        # Check if tool matches
        if disruption.affected_tool and disruption.affected_tool != tool_name:
            return False

        # Check trigger conditions
        if disruption.trigger == DisruptionTrigger.AFTER_N_ACTIONS:
            if self._action_count < disruption.trigger_after_n_actions:
                return False

        # Apply probability
        if random.random() > disruption.probability:
            return False

        return True

    def get_triggered_disruptions(self) -> List[Dict[str, Any]]:
        """Get list of all triggered disruptions."""
        return self._triggered_disruptions.copy()

    def get_disruption_count(self) -> int:
        """Get number of triggered disruptions."""
        return len(self._triggered_disruptions)

    def has_disruptions_configured(self) -> bool:
        """Check if any disruptions are configured."""
        return len(self._disruptions) > 0


def get_disruption_engine() -> DisruptionEngine:
    """Get the singleton disruption engine instance."""
    return DisruptionEngine()
