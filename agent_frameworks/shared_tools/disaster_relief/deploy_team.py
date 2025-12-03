"""Action tool: Deploy a relief team."""

from langchain_core.tools import tool
from typing import Optional, Dict

from ..state_manager import StateManager
from ..disruption_engine import DisruptionEngine
from ..logging_config import get_logger, log_tool_call, log_tool_result

logger = get_logger("realm_bench.tools")


@tool
def deploy_team(
    team_id: str,
    region_id: str,
    mission_type: str,
    personnel_count: int = 1,
    supplies: Optional[Dict[str, int]] = None
) -> str:
    """
    Deploy a relief team to a region.

    This is an ACTION tool that can be compensated by recall_team.

    Args:
        team_id: Unique identifier for the team (e.g., "team_1", "medical_team_A")
        region_id: Region to deploy to (e.g., "region_1", "zone_critical")
        mission_type: Type of mission (e.g., "medical", "logistics", "search_rescue")
        personnel_count: Number of personnel in the team (default: 1)
        supplies: Dictionary of supplies with quantities (optional)

    Returns:
        Status message with deployment details
    """
    state = StateManager()
    disruption = DisruptionEngine()

    # Generate deployment ID
    deployment_id = f"deploy_{team_id}_{region_id}"

    log_tool_call(logger, "deploy_team", {
        "team_id": team_id,
        "region_id": region_id,
        "mission_type": mission_type
    })

    # Check for disruptions
    disruption_error = disruption.check_disruption("deploy_team", {
        "team_id": team_id,
        "region_id": region_id
    })

    if disruption_error:
        log_tool_result(logger, "deploy_team", False, disruption_error)
        return f"FAILED: {disruption_error}"

    # Create deployment data
    deployment_data = {
        "team_id": team_id,
        "region_id": region_id,
        "mission_type": mission_type,
        "personnel_count": personnel_count,
        "supplies": supplies or {},
        "status": "deployed"
    }

    # Try to create deployment
    success = state.deploy_team(deployment_id, deployment_data)

    if success:
        state.record_action("deploy_team", {
            "deployment_id": deployment_id,
            **deployment_data
        })

        supplies_str = ""
        if supplies:
            supplies_list = [f"{v} {k}" for k, v in supplies.items()]
            supplies_str = f" with {', '.join(supplies_list)}"

        result = (
            f"SUCCESS: Team {team_id} ({personnel_count} personnel) deployed to "
            f"{region_id} for {mission_type} mission{supplies_str}. "
            f"Deployment ID: {deployment_id}"
        )
        log_tool_result(logger, "deploy_team", True, result)
        return result
    else:
        result = f"FAILED: Deployment {deployment_id} already exists"
        log_tool_result(logger, "deploy_team", False, result)
        return result
