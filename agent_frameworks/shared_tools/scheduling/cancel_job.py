"""Compensation tool: Cancel a scheduled job."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..logging_config import get_logger, log_tool_call, log_tool_result, log_compensation

logger = get_logger("realm_bench.tools")


@tool
def cancel_job(job_id: str) -> str:
    """
    Cancel a previously scheduled job (compensation action).

    This is a COMPENSATION tool that reverses schedule_job.

    Args:
        job_id: Unique identifier for the job to cancel

    Returns:
        Status message indicating success or failure
    """
    state = StateManager()

    log_tool_call(logger, "cancel_job", {"job_id": job_id})

    # Try to cancel the job
    cancelled_job = state.cancel_job(job_id)

    if cancelled_job:
        state.record_compensation("cancel_job", {
            "job_id": job_id,
            "cancelled_data": {
                "machine_id": cancelled_job.machine_id,
                "start_time": cancelled_job.start_time,
                "end_time": cancelled_job.end_time
            }
        }, original_action="schedule_job")

        log_compensation(logger, "schedule_job", "cancel_job", f"Job {job_id} cancelled")

        result = (
            f"SUCCESS: Job {job_id} cancelled. "
            f"Machine {cancelled_job.machine_id} freed from time "
            f"{cancelled_job.start_time} to {cancelled_job.end_time}"
        )
        log_tool_result(logger, "cancel_job", True, result)
        return result
    else:
        result = f"FAILED: Job {job_id} not found in schedule"
        log_tool_result(logger, "cancel_job", False, result)
        return result
