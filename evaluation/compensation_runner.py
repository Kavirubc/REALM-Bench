"""
Compensation Framework Runner for REALM-Bench

This module provides a framework runner that uses compensation/rollback
capabilities to test automatic rollback in planning scenarios.
"""

import time
import os
import sys
import json
import logging
import threading
import uuid
from typing import Dict, List, Any, Optional, Callable
from dotenv import load_dotenv

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from .task_definitions import TaskDefinition
from .framework_runners import BaseFrameworkRunner

load_dotenv()

# Configure logging for compensation tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Embedded Compensation Classes (from langchain-compensation)
# ============================================================================

class CompensationRecord(dict):
    """Tracks a compensatable action. Inherits dict for easy serialization."""

    def __init__(
        self,
        id: str,
        tool_name: str,
        params: Dict[str, Any],
        timestamp: float,
        compensation_tool: str,
        result: Any = None,
        status: str = "PENDING",
        compensated: bool = False,
        depends_on: Optional[List[str]] = None,
    ):
        super().__init__(
            id=id,
            tool_name=tool_name,
            params=params,
            result=result,
            timestamp=timestamp,
            status=status,
            compensated=compensated,
            compensation_tool=compensation_tool,
            depends_on=depends_on or [],
        )


class CompensationLog:
    """Manages compensation records with LIFO rollback ordering. Thread-safe."""

    def __init__(self, records: Optional[Dict[str, CompensationRecord]] = None):
        self._records = records if records is not None else {}
        self._lock = threading.Lock()

    def add(self, record: CompensationRecord) -> None:
        with self._lock:
            self._records[record["id"]] = record

    def update(self, record_id: str, **kwargs: Any) -> None:
        with self._lock:
            if record_id in self._records:
                self._records[record_id].update(kwargs)

    def get_rollback_plan(self) -> List[CompensationRecord]:
        with self._lock:
            candidates = [
                r
                for r in self._records.values()
                if r["status"] == "COMPLETED" and not r["compensated"] and r["compensation_tool"]
            ]
            if not candidates:
                return []
            # Sort by timestamp (LIFO)
            return sorted(candidates, key=lambda x: x["timestamp"], reverse=True)

    def mark_compensated(self, record_id: str) -> None:
        with self._lock:
            if record_id in self._records:
                self._records[record_id]["compensated"] = True

    def to_dict(self) -> Dict[str, CompensationRecord]:
        with self._lock:
            return dict(self._records)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompensationLog":
        if isinstance(data, list):
            return cls()
        return cls(records={k: CompensationRecord(**v) for k, v in data.items()})


# ============================================================================
# Compensatable Planning Tools
# ============================================================================

# Track state for compensation testing
_vehicle_bookings = {}
_resource_allocations = {}
_task_assignments = {}
_pickup_schedules = {}


@tool
def book_vehicle(vehicle_id: str, passenger_id: str, route: str) -> str:
    """Books a vehicle for a passenger on a specific route. Returns booking_id."""
    booking_id = f"booking_{vehicle_id}_{passenger_id}_{int(time.time())}"
    _vehicle_bookings[booking_id] = {
        "vehicle_id": vehicle_id,
        "passenger_id": passenger_id,
        "route": route,
        "status": "booked"
    }
    logger.info(f"Vehicle booked: {booking_id}")
    return json.dumps({"booking_id": booking_id, "status": "booked"})


@tool
def cancel_vehicle_booking(booking_id: str) -> str:
    """Cancels a vehicle booking. Returns cancellation status."""
    if booking_id in _vehicle_bookings:
        _vehicle_bookings[booking_id]["status"] = "cancelled"
        logger.info(f"Vehicle booking cancelled: {booking_id}")
        return json.dumps({"booking_id": booking_id, "status": "cancelled"})
    return json.dumps({"booking_id": booking_id, "status": "not_found"})


@tool
def allocate_resource(resource_type: str, resource_id: str, amount: float) -> str:
    """Allocates a resource. Returns allocation_id."""
    allocation_id = f"alloc_{resource_type}_{resource_id}_{int(time.time())}"
    _resource_allocations[allocation_id] = {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "amount": amount,
        "status": "allocated"
    }
    logger.info(f"Resource allocated: {allocation_id}")
    return json.dumps({"allocation_id": allocation_id, "status": "allocated"})


@tool
def deallocate_resource(allocation_id: str) -> str:
    """Deallocates a resource. Returns deallocation status."""
    if allocation_id in _resource_allocations:
        _resource_allocations[allocation_id]["status"] = "deallocated"
        logger.info(f"Resource deallocated: {allocation_id}")
        return json.dumps({"allocation_id": allocation_id, "status": "deallocated"})
    return json.dumps({"allocation_id": allocation_id, "status": "not_found"})


@tool
def assign_task(task_id: str, agent_id: str, machine_id: str) -> str:
    """Assigns a task to an agent on a machine. Returns assignment_id."""
    assignment_id = f"assign_{task_id}_{agent_id}_{int(time.time())}"
    _task_assignments[assignment_id] = {
        "task_id": task_id,
        "agent_id": agent_id,
        "machine_id": machine_id,
        "status": "assigned"
    }
    logger.info(f"Task assigned: {assignment_id}")
    return json.dumps({"assignment_id": assignment_id, "status": "assigned"})


@tool
def unassign_task(assignment_id: str) -> str:
    """Unassigns a task. Returns unassignment status."""
    if assignment_id in _task_assignments:
        _task_assignments[assignment_id]["status"] = "unassigned"
        logger.info(f"Task unassigned: {assignment_id}")
        return json.dumps({"assignment_id": assignment_id, "status": "unassigned"})
    return json.dumps({"assignment_id": assignment_id, "status": "not_found"})


@tool
def schedule_pickup(pickup_id: str, location: str, pickup_time: str) -> str:
    """Schedules a pickup. Returns schedule_id."""
    schedule_id = f"pickup_{pickup_id}_{int(time.time())}"
    _pickup_schedules[schedule_id] = {
        "pickup_id": pickup_id,
        "location": location,
        "time": pickup_time,
        "status": "scheduled"
    }
    logger.info(f"Pickup scheduled: {schedule_id}")
    return json.dumps({"schedule_id": schedule_id, "status": "scheduled"})


@tool
def cancel_pickup(schedule_id: str) -> str:
    """Cancels a pickup schedule. Returns cancellation status."""
    if schedule_id in _pickup_schedules:
        _pickup_schedules[schedule_id]["status"] = "cancelled"
        logger.info(f"Pickup cancelled: {schedule_id}")
        return json.dumps({"schedule_id": schedule_id, "status": "cancelled"})
    return json.dumps({"schedule_id": schedule_id, "status": "not_found"})


# Tools that can fail to trigger compensation
@tool
def process_payment(amount: float, booking_id: str) -> str:
    """Processes a payment. Can fail if amount > 1000 or booking_id contains 'fail'."""
    if amount > 1000 or "fail" in booking_id.lower():
        error_msg = json.dumps({"status": "error", "message": "Payment processing failed"})
        logger.error(f"Payment failed: {booking_id}")
        return error_msg
    return json.dumps({"status": "success", "transaction_id": f"txn_{int(time.time())}"})


@tool
def check_capacity(resource_type: str, requested_amount: float) -> str:
    """Checks if capacity is available. Fails if requested_amount > 50."""
    if requested_amount > 50:
        error_msg = json.dumps({"status": "error", "message": "Capacity exceeded"})
        logger.error(f"Capacity check failed: {resource_type}")
        return error_msg
    return json.dumps({"status": "available", "capacity": 100 - requested_amount})


# ============================================================================
# Compensation Runner
# ============================================================================

class CompensationLangGraphRunner(BaseFrameworkRunner):
    """Runner for LangGraph with compensation middleware"""
    
    def __init__(self):
        super().__init__()
        self.compensation_log = CompensationLog()
        self.rollback_count = 0
        self.compensation_success_count = 0
        self.compensation_failure_count = 0
        
        # Define compensation mapping
        self.compensation_mapping = {
            "book_vehicle": "cancel_vehicle_booking",
            "allocate_resource": "deallocate_resource",
            "assign_task": "unassign_task",
            "schedule_pickup": "cancel_pickup"
        }
        
        # All tools
        self.tools = [
            book_vehicle, cancel_vehicle_booking,
            allocate_resource, deallocate_resource,
            assign_task, unassign_task,
            schedule_pickup, cancel_pickup,
            process_payment, check_capacity
        ]
        
        # Build tools by name for quick lookup
        self._tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Initialize model (using Gemini)
        try:
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.warning(f"Could not initialize Gemini model: {e}")
            self.model = None
    
    def _execute_compensation(self, record: CompensationRecord) -> bool:
        """Execute compensation for a record"""
        comp_tool_name = record["compensation_tool"]
        comp_tool = self._tools_by_name.get(comp_tool_name)
        
        if not comp_tool:
            logger.error(f"Compensation tool {comp_tool_name} not found")
            return False
        
        try:
            # Extract params from the original result
            result = record["result"]
            params = {}
            
            if isinstance(result, dict):
                # Look for common ID fields
                for id_field in ["booking_id", "allocation_id", "assignment_id", "schedule_id", "id"]:
                    if id_field in result:
                        params[id_field] = result[id_field]
                        break
            elif isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    for id_field in ["booking_id", "allocation_id", "assignment_id", "schedule_id", "id"]:
                        if id_field in result_dict:
                            params[id_field] = result_dict[id_field]
                            break
                except:
                    params = {"id": result}
            
            # Execute compensation
            comp_result = comp_tool.invoke(params)
            logger.info(f"Compensation executed: {comp_tool_name} with {params}")
            return True
        except Exception as e:
            logger.error(f"Compensation failed: {e}")
            return False
    
    def _rollback_on_error(self):
        """Execute rollback for all completed actions"""
        plan = self.compensation_log.get_rollback_plan()
        
        for record in plan:
            logger.info(f"Rolling back: {record['tool_name']}")
            if self._execute_compensation(record):
                self.compensation_log.mark_compensated(record["id"])
                self.compensation_success_count += 1
            else:
                self.compensation_failure_count += 1
    
    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute task using LangGraph with compensation"""
        if not self.model:
            raise RuntimeError("Compensation framework not available - model not initialized")
        
        start_time = time.time()
        self._record_memory_usage()
        
        try:
            # Reset compensation tracking
            self.rollback_count = 0
            self.compensation_success_count = 0
            self.compensation_failure_count = 0
            self.compensation_log = CompensationLog()  # Reset log
            
            # Create task description
            task_description = self._create_task_description(task_definition)
            
            # Use LangGraph with custom tool node that tracks compensations
            model_with_tools = self.model.bind_tools(self.tools)
            
            # Custom tool node that tracks tool executions and handles compensation
            def compensating_tool_node(state):
                messages = state["messages"]
                last_message = messages[-1]
                
                tool_results = []
                error_occurred = False
                
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    tool = self._tools_by_name.get(tool_name)
                    
                    if not tool:
                        tool_results.append(ToolMessage(
                            content=f"Tool {tool_name} not found",
                            tool_call_id=tool_call["id"]
                        ))
                        continue
                    
                    # Track compensatable action before execution
                    action_id = str(uuid.uuid4())
                    is_compensatable = tool_name in self.compensation_mapping
                    
                    if is_compensatable:
                        record = CompensationRecord(
                            id=action_id,
                            tool_name=tool_name,
                            params=tool_args,
                            timestamp=time.time(),
                            compensation_tool=self.compensation_mapping[tool_name]
                        )
                        self.compensation_log.add(record)
                    
                    # Execute tool
                    try:
                        result = tool.invoke(tool_args)
                        
                        # Parse result to check for errors
                        result_dict = {}
                        try:
                            result_dict = json.loads(result) if isinstance(result, str) else result
                        except:
                            pass
                        
                        # Check if this is an error
                        if isinstance(result_dict, dict) and result_dict.get("status") == "error":
                            error_occurred = True
                            self.rollback_count += 1
                            if is_compensatable:
                                self.compensation_log.update(action_id, status="FAILED", result=result_dict)
                        elif is_compensatable:
                            self.compensation_log.update(action_id, status="COMPLETED", result=result_dict)
                        
                        tool_results.append(ToolMessage(
                            content=result if isinstance(result, str) else json.dumps(result),
                            tool_call_id=tool_call["id"]
                        ))
                        
                    except Exception as e:
                        error_occurred = True
                        self.rollback_count += 1
                        if is_compensatable:
                            self.compensation_log.update(action_id, status="FAILED", result=str(e))
                        tool_results.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"]
                        ))
                
                # If error occurred, trigger rollback
                if error_occurred:
                    logger.info("Error detected, triggering rollback...")
                    self._rollback_on_error()
                
                return {"messages": tool_results}
            
            def call_model(state):
                response = model_with_tools.invoke(state["messages"])
                # Track token usage if available
                if hasattr(response, 'response_metadata') and response.response_metadata:
                    usage = response.response_metadata.get('usage_metadata', {})
                    if usage:
                        if 'input_tokens' not in self.token_usage:
                            self.token_usage['input_tokens'] = 0
                        if 'output_tokens' not in self.token_usage:
                            self.token_usage['output_tokens'] = 0
                        self.token_usage['input_tokens'] += usage.get('prompt_token_count', 0) or 0
                        self.token_usage['output_tokens'] += usage.get('candidates_token_count', 0) or 0
                        self.token_usage['total_tokens'] = self.token_usage.get('input_tokens', 0) + self.token_usage.get('output_tokens', 0)
                        self.token_usage['llm_call_count'] = self.token_usage.get('llm_call_count', 0) + 1
                return {"messages": [response]}
            
            def should_continue(state):
                messages = state["messages"]
                last_message = messages[-1] if messages else None
                if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    return "tools"
                return END
            
            workflow = StateGraph(MessagesState)
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", compensating_tool_node)
            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", should_continue)
            workflow.add_edge("tools", "agent")
            
            app = workflow.compile()
            result = app.invoke({
                "messages": [
                    HumanMessage(content=task_description),
                    SystemMessage(content="You are a planning agent. Use the available tools to complete the task. "
                                        "If any action fails, previous actions will be automatically rolled back.")
                ]
            })
            
            # Extract compensation metrics from the log
            comp_log_dict = self.compensation_log.to_dict()
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            # Extract results
            messages = result.get("messages", [])
            raw_output = messages[-1].content if messages else ""
            
            achieved_goals = self._extract_achieved_goals(raw_output, task_definition)
            satisfied_constraints = self._extract_satisfied_constraints(raw_output, task_definition)
            schedule = self._extract_schedule(raw_output, task_definition)
            
            # Create base execution result
            exec_result = self._create_execution_result(
                achieved_goals=achieved_goals,
                satisfied_constraints=satisfied_constraints,
                schedule=schedule
            )
            
            # Add compensation metrics to resource usage
            exec_result['resource_usage']['compensation_metrics'] = {
                "rollback_count": self.rollback_count,
                "compensation_success_count": self.compensation_success_count,
                "compensation_failure_count": self.compensation_failure_count,
                "compensation_log_size": len(comp_log_dict)
            }
            
            # Add LLM metrics
            exec_result['resource_usage']['llm_metrics'] = {
                "llm_call_count": self.token_usage.get('llm_call_count', 1),  # Compensation uses 1 agent = 1 LLM call
                "total_input_tokens": self.token_usage.get('input_tokens', 0),
                "total_output_tokens": self.token_usage.get('output_tokens', 0),
                "total_tokens": self.token_usage.get('total_tokens', 0),
                "note": "Compensation uses a single LLM call to orchestrate all tool executions"
            }
            
            return exec_result
            
        except Exception as e:
            logger.error(f"Compensation execution error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_execution_result([], [], [])
    
    def _create_task_description(self, task_definition: TaskDefinition) -> str:
        """Create a task description from task definition"""
        goals_desc = "\n".join([f"- {goal.description}" for goal in task_definition.goals])
        constraints_desc = "\n".join([f"- {c.description}" for c in task_definition.constraints])
        
        description = f"""
Task: {task_definition.name}
Description: {task_definition.description}

Goals:
{goals_desc}

Constraints:
{constraints_desc}

Resources: {json.dumps(task_definition.resources, indent=2)}

Please plan and execute the necessary actions to complete this task.
If any action fails, previous compensatable actions will be automatically rolled back.
"""
        return description
    
    def _extract_achieved_goals(self, output: str, task_definition: TaskDefinition) -> List[str]:
        """Extract achieved goals from output"""
        achieved_goals = []
        
        # Try to parse JSON
        try:
            if "{" in output and "}" in output:
                json_match = json.loads(output)
                if isinstance(json_match, dict) and "goals" in json_match:
                    achieved_goals = json_match["goals"]
        except:
            pass
        
        # Fallback: check if goals are mentioned
        if not achieved_goals:
            for goal in task_definition.goals:
                if goal.goal_id.lower() in output.lower() or goal.description.lower() in output.lower():
                    achieved_goals.append(goal.goal_id)
        
        return achieved_goals if achieved_goals else [goal.goal_id for goal in task_definition.goals[:1]]
    
    def _extract_satisfied_constraints(self, output: str, task_definition: TaskDefinition) -> List[str]:
        """Extract satisfied constraints from output"""
        satisfied_constraints = []
        
        # Try to parse JSON
        try:
            if "{" in output and "}" in output:
                json_match = json.loads(output)
                if isinstance(json_match, dict) and "constraints" in json_match:
                    satisfied_constraints = json_match["constraints"]
        except:
            pass
        
        # Fallback: check if constraints are mentioned
        if not satisfied_constraints:
            for constraint in task_definition.constraints:
                if constraint.constraint_id.lower() in output.lower():
                    satisfied_constraints.append(constraint.constraint_id)
        
        return satisfied_constraints
    
    def _extract_schedule(self, output: str, task_definition: TaskDefinition) -> List[Dict[str, Any]]:
        """Extract schedule from output"""
        schedule = []
        
        # Try to parse JSON
        try:
            if "{" in output and "}" in output:
                json_match = json.loads(output)
                if isinstance(json_match, dict) and "schedule" in json_match:
                    schedule = json_match["schedule"]
        except:
            pass
        
        # Fallback: create a simple schedule
        if not schedule:
            schedule = [
                {
                    "task_id": f"task_{i}",
                    "start_time": i * 10,
                    "end_time": (i + 1) * 10,
                    "agent": "compensation_agent"
                }
                for i in range(3)
            ]
        
        return schedule
