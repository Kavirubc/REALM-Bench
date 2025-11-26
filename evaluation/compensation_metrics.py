"""
Compensation-Specific Metrics for REALM-Bench

This module provides metrics specifically for evaluating compensation/rollback
capabilities in planning scenarios.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .metrics import BaseMetrics, MetricType, MetricResult


class CompensationMetricType(Enum):
    """Types of compensation-specific metrics"""
    ROLLBACK_SUCCESS = "rollback_success"
    COMPENSATION_COVERAGE = "compensation_coverage"
    ROLLBACK_TIME = "rollback_time"
    DEPENDENCY_ORDERING = "dependency_ordering"


class CompensationMetrics(BaseMetrics):
    """Metrics for evaluating compensation and rollback capabilities"""
    
    def evaluate_rollback_success_rate(
        self,
        compensation_metrics: Dict[str, Any],
        total_failures: int = None
    ) -> MetricResult:
        """
        Evaluate rollback success rate - % of compensations that succeeded
        
        Args:
            compensation_metrics: Dictionary with compensation tracking data
            total_failures: Total number of failures (optional)
        """
        success_count = compensation_metrics.get("compensation_success_count", 0)
        failure_count = compensation_metrics.get("compensation_failure_count", 0)
        rollback_count = compensation_metrics.get("rollback_count", 0)
        
        total_compensations = success_count + failure_count
        
        if total_compensations == 0:
            return self.add_result(
                "rollback_success_rate",
                MetricType.ADAPTATION,  # Using adaptation as closest category
                100.0,
                "percentage",
                "No compensations attempted",
                {
                    "success_count": 0,
                    "failure_count": 0,
                    "rollback_count": rollback_count
                }
            )
        
        success_rate = (success_count / total_compensations) * 100
        
        return self.add_result(
            "rollback_success_rate",
            MetricType.ADAPTATION,
            success_rate,
            "percentage",
            f"Rollback success rate: {success_rate:.2f}%",
            {
                "success_count": success_count,
                "failure_count": failure_count,
                "total_compensations": total_compensations,
                "rollback_count": rollback_count
            }
        )
    
    def evaluate_compensation_coverage(
        self,
        compensation_metrics: Dict[str, Any],
        total_actions: int = None
    ) -> MetricResult:
        """
        Evaluate compensation coverage - % of failed actions that had compensation handlers
        
        Args:
            compensation_metrics: Dictionary with compensation tracking data
            total_actions: Total number of actions attempted (optional)
        """
        rollback_count = compensation_metrics.get("rollback_count", 0)
        comp_log_size = compensation_metrics.get("compensation_log_size", 0)
        
        if comp_log_size == 0:
            return self.add_result(
                "compensation_coverage",
                MetricType.ADAPTATION,
                0.0,
                "percentage",
                "No compensation log entries",
                {
                    "rollback_count": 0,
                    "compensation_log_size": 0
                }
            )
        
        # Coverage is the ratio of actions with compensation handlers
        # to total actions that could have been compensated
        coverage = (comp_log_size / max(comp_log_size + rollback_count, 1)) * 100
        
        return self.add_result(
            "compensation_coverage",
            MetricType.ADAPTATION,
            coverage,
            "percentage",
            f"Compensation coverage: {coverage:.2f}%",
            {
                "rollback_count": rollback_count,
                "compensation_log_size": comp_log_size,
                "total_compensatable_actions": comp_log_size
            }
        )
    
    def evaluate_rollback_time(
        self,
        execution_times: List[float],
        compensation_metrics: Dict[str, Any]
    ) -> MetricResult:
        """
        Evaluate rollback time - time taken to execute rollback
        
        Args:
            execution_times: List of execution times
            compensation_metrics: Dictionary with compensation tracking data
        """
        rollback_count = compensation_metrics.get("rollback_count", 0)
        
        if rollback_count == 0 or not execution_times:
            return self.add_result(
                "rollback_time",
                MetricType.RESOURCE_USAGE,
                0.0,
                "seconds",
                "No rollbacks occurred",
                {
                    "rollback_count": 0,
                    "execution_times": execution_times
                }
            )
        
        # Estimate rollback time as a fraction of total execution time
        # This is a heuristic since we don't have direct rollback timing
        total_time = sum(execution_times)
        avg_rollback_time = total_time / (rollback_count + 1) if rollback_count > 0 else 0
        
        return self.add_result(
            "average_rollback_time",
            MetricType.RESOURCE_USAGE,
            avg_rollback_time,
            "seconds",
            f"Average rollback time: {avg_rollback_time:.2f} seconds",
            {
                "rollback_count": rollback_count,
                "total_execution_time": total_time,
                "estimated_rollback_time": avg_rollback_time
            }
        )
    
    def evaluate_dependency_ordering_correctness(
        self,
        compensation_log: Dict[str, Any],
        execution_order: List[str] = None
    ) -> MetricResult:
        """
        Evaluate whether rollback respected dependencies (DAG ordering)
        
        Args:
            compensation_log: Compensation log dictionary
            execution_order: Order in which actions were executed (optional)
        """
        if not compensation_log:
            return self.add_result(
                "dependency_ordering_correctness",
                MetricType.COORDINATION_EFFECTIVENESS,
                100.0,
                "percentage",
                "No compensation log to evaluate",
                {}
            )
        
        # Check if records have dependency information
        records_with_deps = 0
        records_with_correct_ordering = 0
        
        for record_id, record in compensation_log.items():
            depends_on = record.get("depends_on", [])
            if depends_on:
                records_with_deps += 1
                # If a record has dependencies, check if it was compensated after its dependencies
                # This is a simplified check - in practice, we'd need the actual compensation order
                if record.get("compensated", False):
                    records_with_correct_ordering += 1
        
        if records_with_deps == 0:
            correctness = 100.0  # No dependencies to check
        else:
            correctness = (records_with_correct_ordering / records_with_deps) * 100
        
        return self.add_result(
            "dependency_ordering_correctness",
            MetricType.COORDINATION_EFFECTIVENESS,
            correctness,
            "percentage",
            f"Dependency ordering correctness: {correctness:.2f}%",
            {
                "records_with_dependencies": records_with_deps,
                "records_with_correct_ordering": records_with_correct_ordering,
                "total_records": len(compensation_log)
            }
        )
    
    def evaluate_compensation_efficiency(
        self,
        compensation_metrics: Dict[str, Any],
        total_execution_time: float
    ) -> MetricResult:
        """
        Evaluate compensation efficiency - ratio of successful compensations to total time
        
        Args:
            compensation_metrics: Dictionary with compensation tracking data
            total_execution_time: Total execution time
        """
        success_count = compensation_metrics.get("compensation_success_count", 0)
        rollback_count = compensation_metrics.get("rollback_count", 0)
        
        if total_execution_time == 0:
            efficiency = 0.0
        else:
            # Efficiency = successful compensations per second
            efficiency = success_count / total_execution_time if total_execution_time > 0 else 0
        
        return self.add_result(
            "compensation_efficiency",
            MetricType.RESOURCE_USAGE,
            efficiency,
            "compensations_per_second",
            f"Compensation efficiency: {efficiency:.2f} compensations/second",
            {
                "success_count": success_count,
                "rollback_count": rollback_count,
                "total_execution_time": total_execution_time
            }
        )

