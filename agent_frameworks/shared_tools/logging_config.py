"""
Minimal logging configuration for REALM-Bench.

Provides console and JSON file loggin. 
"""

import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Clean console formatter without emojis."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_benchmark_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    logger_name: str = "realm_bench"
) -> logging.Logger:
    """
    Configure logging for benchmark execution.

    Args:
        log_file: Path to JSON log file. If None, only console logging.
        level: Logging level (default INFO).
        logger_name: Name of the logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    # File handler (JSON format)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "realm_bench") -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


def log_tool_call(logger: logging.Logger, tool_name: str, args: Dict[str, Any]) -> None:
    """Log a tool call."""
    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    logger.info(f"TOOL_CALL: {tool_name}({args_str})")


def log_tool_result(
    logger: logging.Logger,
    tool_name: str,
    success: bool,
    result: str
) -> None:
    """Log a tool result."""
    status = "SUCCESS" if success else "FAILED"
    # Truncate long results
    result_display = result[:200] + "..." if len(result) > 200 else result
    logger.info(f"TOOL_RESULT: {tool_name} - {status}: {result_display}")


def log_compensation(
    logger: logging.Logger,
    action_tool: str,
    comp_tool: str,
    reason: str
) -> None:
    """Log a compensation event."""
    logger.info(f"COMPENSATION: {comp_tool} compensates {action_tool} - {reason}")


def log_disruption(
    logger: logging.Logger,
    disruption_type: str,
    affected_tool: str,
    details: Optional[str] = None
) -> None:
    """Log a disruption event."""
    msg = f"DISRUPTION: {disruption_type} affecting {affected_tool}"
    if details:
        msg += f" - {details}"
    logger.info(msg)


def log_task_start(logger: logging.Logger, task_id: str, framework: str) -> None:
    """Log task execution start."""
    logger.info(f"TASK_START: {task_id} using {framework}")


def log_task_end(
    logger: logging.Logger,
    task_id: str,
    framework: str,
    success: bool,
    duration: float
) -> None:
    """Log task execution end."""
    status = "completed" if success else "failed"
    logger.info(f"TASK_END: {task_id} ({framework}) {status} in {duration:.2f}s")
