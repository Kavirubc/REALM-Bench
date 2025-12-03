"""Compensation tool: Recall a deployed team."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..logging_config import get_logger, log_tool_call, log_tool_result, log_compensation

logger = get_logger("realm_bench.tools")


@tool
def recall_team(deployment_id: str) -> str:
    """
    Recall a deployed team from their mission (compensation action).

    This is a COMPENSATION tool that reverses deploy_team.

    Args:
        deployment_id: Unique identifier for the deployment
            (format: "deploy_{team_id}_{region_id}")

    Returns:
        Status message indicating success or failure
    """
    state = StateManager()

    log_tool_call(logger, "recall_team", {"deployment_id": deployment_id})

    # Try to recall the team
    cancelled = state.recall_team(deployment_id)

    if cancelled:
        state.record_compensation("recall_team", {
            "deployment_id": deployment_id,
            "cancelled_data": {
                "team_id": cancelled.team_id,
                "region_id": cancelled.region_id,
                "mission_type": cancelled.mission_type
            }
        }, original_action="deploy_team")

        log_compensation(
            logger,
            "deploy_team",
            "recall_team",
            f"Deployment {deployment_id} recalled"
        )

        result = (
            f"SUCCESS: Team {cancelled.team_id} recalled from {cancelled.region_id}. "
            f"Team and resources are now available for redeployment"
        )
        log_tool_result(logger, "recall_team", True, result)
        return result
    else:
        result = f"FAILED: Deployment {deployment_id} not found"
        log_tool_result(logger, "recall_team", False, result)
        return result
