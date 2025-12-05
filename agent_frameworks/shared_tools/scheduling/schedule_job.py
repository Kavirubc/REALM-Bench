"""Tool to schedule a job on a machine."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..disruption_engine import DisruptionEngine
from ..logging_config import get_logger, log_tool_call, log_tool_result

logger = get_logger("realm_bench.tools")


@tool
def schedule_job(
    job_id: str,
    machine_id: str,
    start_time: int,
    duration: int,
    priority: int = 1
) -> dict:
    """
    Schedule a job on a specific machine at a given time slot.

    Args:
        job_id: Unique identifier for the job (e.g., "job_1", "J1")
        machine_id: Machine to schedule the job on (e.g., "machine_1", "M1")
        start_time: Start time for the job (integer time unit)
        duration: Duration of the job (integer time units)
        priority: Job priority, 1 is highest (default: 1)

    Returns:
        Dict with status and job details
    """
    state = StateManager()
    disruption = DisruptionEngine()

    log_tool_call(logger, "schedule_job", {
        "job_id": job_id,
        "machine_id": machine_id,
        "start_time": start_time,
        "duration": duration
    })

    # Check for disruptions
    disruption_error = disruption.check_disruption("schedule_job", {
        "job_id": job_id,
        "machine_id": machine_id
    })

    if disruption_error:
        log_tool_result(logger, "schedule_job", False, disruption_error)
        return {"status": "error", "error": disruption_error, "job_id": job_id}

    # Create job data
    job_data = {
        "job_id": job_id,
        "machine_id": machine_id,
        "start_time": start_time,
        "end_time": start_time + duration,
        "duration": duration,
        "priority": priority,
        "status": "scheduled"
    }

    # Try to schedule the job
    success = state.schedule_job(job_id, job_data)

    if success:
        state.record_action("schedule_job", job_data)
        log_tool_result(logger, "schedule_job", True, f"Job {job_id} scheduled")
        return {
            "status": "success",
            "job_id": job_id,
            "machine_id": machine_id,
            "start_time": start_time,
            "end_time": start_time + duration
        }
    else:
        log_tool_result(logger, "schedule_job", False, f"Job {job_id} already exists")
        return {"status": "error", "error": f"Job {job_id} already exists in schedule", "job_id": job_id}
