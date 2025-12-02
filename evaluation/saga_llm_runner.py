"""
SagaLLM Framework Runner for REALM-Bench P1-P11 Tasks

This module provides a runner for SagaLLM framework to compare against
langchain-compensation on all 11 REALM-Bench planning scenarios.
"""

import time
import os
import sys
import json
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "realm-bench-sagallm")

saga_lib_path = os.path.join(project_root, "agent_frameworks", "sagallm_lib")
if saga_lib_path not in sys.path:
    sys.path.insert(0, saga_lib_path)

from evaluation.task_definitions import TaskDefinition
from evaluation.framework_runners import BaseFrameworkRunner
from evaluation.planning_tools import (
    TASK_TOOLS, COMPENSATION_MAPPING, reset_state
)
from langchain_core.tools import StructuredTool

try:
    from multi_agent.saga import Saga
    from multi_agent.agent import Agent
    SAGA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SagaLLM not available: {e}")
    SAGA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"


class CompensatableSagaAgent(Agent):
    """Extended Saga Agent with actual tool compensation."""

    def __init__(self, name, backstory, task_description, tools=None, compensation_tool=None):
        saga_tools = []
        if tools:
            for tool in tools:
                if isinstance(tool, StructuredTool):
                    from tool_agent.tool import Tool as SagaTool
                    properties = {}
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        schema_props = tool.args_schema.schema().get("properties", {})
                        for param_name, param_info in schema_props.items():
                            param_type = param_info.get("type", "string")
                            type_mapping = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}
                            properties[param_name] = {"type": type_mapping.get(param_type, "str")}

                    saga_tool = SagaTool(
                        name=tool.name,
                        fn=tool.func,
                        fn_signature=json.dumps({
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {"properties": properties}
                        })
                    )
                    saga_tools.append(saga_tool)
                else:
                    saga_tools.append(tool)

        from utils.gemini_client import GeminiClientWrapper
        self._gemini_client = GeminiClientWrapper(model="gemini-2.0-flash", temperature=0)

        super().__init__(
            name=name,
            backstory=backstory,
            task_description=task_description,
            tools=saga_tools,
            llm="gemini-2.0-flash"
        )

        from planning_agent.react_agent import ReactAgent

        class ToolTrackingReactAgent(ReactAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_tool_results = []
                if not hasattr(self, 'tools_dict'):
                    self.tools_dict = {tool.name: tool for tool in self.tools}

            def process_tool_calls(self, tool_calls_content: list) -> dict:
                try:
                    observations = super().process_tool_calls(tool_calls_content)
                    for tool_call_str in tool_calls_content:
                        try:
                            tool_call = json.loads(tool_call_str) if isinstance(tool_call_str, str) else tool_call_str
                            tool_name = tool_call.get("name", "unknown")
                            tool_id = tool_call.get("id", 0)
                            result = observations.get(tool_id, "")
                            self.last_tool_results.append({"tool_name": tool_name, "result": str(result)})
                        except Exception as e:
                            logger.warning(f"Error tracking tool result: {e}")
                    return observations
                except Exception as e:
                    logger.error(f"Error in process_tool_calls: {e}")
                    raise

        self.react_agent = ToolTrackingReactAgent(
            tools=saga_tools,
            model="gemini-2.0-flash",
            system_prompt=backstory,
            client=self._gemini_client
        )

        self.compensation_tool = compensation_tool
        self.execution_result = None

    def run(self):
        print(f"🚀 {self.name} executing...")
        try:
            if hasattr(self.react_agent, 'last_tool_results'):
                self.react_agent.last_tool_results = []

            user_message = self.task_description
            output = self.react_agent.run(user_msg=user_message, max_rounds=5)

            if hasattr(self.react_agent, 'last_tool_results') and self.react_agent.last_tool_results:
                self.execution_result = self.react_agent.last_tool_results[-1]["result"]
                result_str = str(self.execution_result)
            else:
                self.execution_result = output
                result_str = output

            if "error" in result_str.lower():
                try:
                    result_data = json.loads(result_str)
                    if result_data.get("status") == "error":
                        raise Exception(f"Tool execution failed: {result_str}")
                except json.JSONDecodeError:
                    if "error" in result_str.lower():
                        raise Exception(f"Tool execution failed: {result_str}")

            return result_str

        except Exception as e:
            self.execution_result = str(e)
            logger.error(f"Error in {self.name}: {e}")
            raise e

    def rollback(self):
        print(f"🔄 Rolling back {self.name}'s operation...")
        if self.compensation_tool and self.execution_result:
            try:
                result_str = str(self.execution_result)
                try:
                    result_data = json.loads(result_str)
                except (json.JSONDecodeError, TypeError):
                    result_data = {}

                comp_args = {}
                for id_field in ["visit_id", "assignment_id", "booking_id", "schedule_id",
                                "deployment_id", "allocation_id", "order_id"]:
                    if id_field in result_data:
                        comp_args[id_field] = result_data[id_field]
                        break

                if comp_args:
                    if hasattr(self.compensation_tool, 'func'):
                        res = self.compensation_tool.func(**comp_args)
                    else:
                        res = "Compensation tool not callable"
                    print(f"✅ Compensation executed: {res}")
            except Exception as e:
                print(f"❌ Compensation failed: {e}")


# Agent configurations for each task
TASK_AGENTS = {
    "P1": [
        {"name": "Location Planner", "task": "Plan optimal route visiting Library, Student Center, Engineering, Sports Complex"},
        {"name": "Time Validator", "task": "Check all locations are open at planned times (9AM-5PM)"},
        {"name": "Tour Executor", "task": "Execute visits to all locations in planned order"},
    ],
    "P2": [
        {"name": "Guide Coordinator", "task": "Check availability of 3 tour guides"},
        {"name": "Group Assigner", "task": "Assign guides to 4 visitor groups"},
        {"name": "Schedule Validator", "task": "Validate no guide has more than 2 groups"},
    ],
    "P3": [
        {"name": "Vehicle Manager", "task": "Check capacity of 3 vehicles (4 passengers each)"},
        {"name": "Ride Planner", "task": "Plan rides for 5 passengers with optimal grouping"},
        {"name": "Booking Agent", "task": "Book all rides and handle capacity issues"},
    ],
    "P4": [
        {"name": "Route Planner", "task": "Plan initial routes for all rides"},
        {"name": "Disruption Handler", "task": "Monitor and handle route disruptions"},
        {"name": "Rerouting Agent", "task": "Update routes when disruptions occur"},
    ],
    "P5": [
        {"name": "Venue Agent", "task": "Book venue (grand_hall) for 200 guests"},
        {"name": "Caterer Agent", "task": "Book catering (gourmet_catering) for 200 guests"},
        {"name": "Entertainment Agent", "task": "Book band (jazz_band) for 100 audience"},
    ],
    "P6": [
        {"name": "Pickup Coordinator", "task": "Schedule airport pickups for arriving family"},
        {"name": "Kitchen Planner", "task": "Assign cooking tasks to family members"},
        {"name": "Capacity Checker", "task": "Check kitchen can handle all cooks"},
    ],
    "P7": [
        {"name": "Team Deployer", "task": "Deploy medical and rescue teams to regions"},
        {"name": "Supply Allocator", "task": "Allocate supplies to affected regions"},
        {"name": "Resource Validator", "task": "Validate supply availability"},
    ],
    "P8": [
        {"name": "Route Checker", "task": "Check wedding transport routes for closures"},
        {"name": "Transport Booker", "task": "Book wedding transport avoiding blocked routes"},
        {"name": "Venue Coordinator", "task": "Book reception venue"},
    ],
    "P9": [
        {"name": "Flight Monitor", "task": "Check flight statuses for delays"},
        {"name": "Pickup Scheduler", "task": "Schedule/reschedule pickups based on flights"},
        {"name": "Dinner Coordinator", "task": "Coordinate cooking around arrivals"},
    ],
    "P10": [
        {"name": "Procurement Agent", "task": "Check supplier capacities and place orders"},
        {"name": "Order Manager", "task": "Manage component orders from suppliers"},
        {"name": "Assembly Scheduler", "task": "Schedule assembly at facilities"},
    ],
    "P11": [
        {"name": "Machine Monitor", "task": "Check machine availability (machine_1 may be down)"},
        {"name": "Job Scheduler", "task": "Schedule jobs on available machines"},
        {"name": "Conflict Resolver", "task": "Resolve scheduling conflicts"},
    ],
}


class SagaLLMRunner(BaseFrameworkRunner):
    """Runner for SagaLLM framework on P1-P11 tasks."""

    def __init__(self):
        super().__init__()
        self.saga = None

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        if not SAGA_AVAILABLE:
            raise RuntimeError("SagaLLM not installed")

        start_time = time.time()
        self._record_memory_usage()
        reset_state()

        task_id = task_definition.task_id
        logger.info(f"Running SagaLLM on task {task_id}")

        try:
            self.saga = Saga()
            agents = self._create_agents_for_task(task_id)

            if not agents:
                raise RuntimeError(f"No agents configured for task {task_id}")

            self.saga.transaction_manager(agents)

            execution_successful = False
            try:
                self.saga.saga_coordinator(with_rollback=True)
                execution_successful = len(self.saga.context) == len(agents)
            except Exception as e:
                logger.warning(f"Saga coordinator exception: {e}")
                execution_successful = False

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()

            # Calculate metrics
            total_goals = len(task_definition.goals)
            achieved = len(self.saga.context) if execution_successful else 0
            goal_satisfaction = achieved / total_goals if total_goals > 0 else 0

            exec_result = self._create_execution_result(
                achieved_goals=[g.goal_id for g in task_definition.goals[:achieved]],
                satisfied_constraints=[c.constraint_id for c in task_definition.constraints],
                schedule=[]
            )

            exec_result['resource_usage']['llm_metrics'] = {
                "llm_call_count": len(agents),
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "note": f"SagaLLM makes {len(agents)} LLM calls (one per agent)"
            }
            exec_result['resource_usage']['execution_time'] = execution_time
            exec_result['resource_usage']['compensation_metrics'] = {
                "rollback_triggered": not execution_successful,
                "actions_compensated": len(agents) - 1 if not execution_successful else 0,
                "compensation_success": not execution_successful,
            }
            exec_result['metrics'] = {
                "goal_satisfaction_rate": goal_satisfaction,
                "execution_time": execution_time,
            }

            return exec_result

        except Exception as e:
            logger.error(f"SagaLLM execution failed: {e}")
            import traceback
            traceback.print_exc()

            execution_time = time.time() - start_time
            error_result = self._create_execution_result([], [], [])
            error_result['resource_usage']['llm_metrics'] = {
                "llm_call_count": 0,
                "total_tokens": 0,
            }
            error_result['resource_usage']['execution_time'] = execution_time
            error_result['resource_usage']['compensation_metrics'] = {
                "rollback_triggered": False,
                "actions_compensated": 0,
                "compensation_success": False,
            }
            return error_result

    def _create_agents_for_task(self, task_id: str) -> List[CompensatableSagaAgent]:
        """Create agents for a specific task."""
        agent_configs = TASK_AGENTS.get(task_id, [])
        tools = TASK_TOOLS.get(task_id, [])

        agents = []
        for i, config in enumerate(agent_configs):
            # Find compensation tool for this agent
            comp_tool = None
            for tool in tools:
                if tool.name in COMPENSATION_MAPPING.values():
                    comp_tool = tool
                    break

            agent = CompensatableSagaAgent(
                name=config["name"],
                backstory=f"You are the {config['name']} for {task_id}",
                task_description=config["task"],
                tools=tools,
                compensation_tool=comp_tool
            )

            # Add dependencies (sequential)
            if i > 0:
                agent.add_dependency(agents[i-1])

            agents.append(agent)

        return agents
