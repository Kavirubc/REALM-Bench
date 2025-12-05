"""Tool to cancel a scheduled job."""

from langchain_core.tools import tool

from ..state_manager import StateManager
from ..logging_config import get_logger, log_tool_call, log_tool_result, log_compensation

logger = get_logger("realm_bench.tools")


@tool
def cancel_job(job_id: str) -> dict:
    """
    Cancel a scheduled job and free up the machine time slot.

    Args:
        job_id: Unique identifier for the job to cancel

    Returns:
        Dict with status and cancellation details
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
        log_tool_result(logger, "cancel_job", True, f"Job {job_id} cancelled")

        return {
            "status": "success",
            "job_id": job_id,
            "machine_id": cancelled_job.machine_id,
            "freed_start": cancelled_job.start_time,
            "freed_end": cancelled_job.end_time
        }
    else:
        log_tool_result(logger, "cancel_job", False, f"Job {job_id} not found")
        return {"status": "error", "error": f"Job {job_id} not found in schedule", "job_id": job_id}
