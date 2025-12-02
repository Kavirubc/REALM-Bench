"""
Compensation Framework Runner for REALM-Bench P1-P11 Tasks

Uses langchain-compensation library to run all 11 REALM-Bench planning scenarios.
Benchmarked against SagaLLM's multi-agent Saga pattern.
"""

import os
import sys
import time
import json
import logging
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "realm-bench-compensation-lib")

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent
from .task_definitions import TaskDefinition
from .framework_runners import BaseFrameworkRunner
from .planning_tools import (
    ALL_TOOLS, COMPENSATION_MAPPING, TASK_TOOLS,
    reset_state
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Task prompts for P1-P11
TASK_PROMPTS = {
    "P1": """You are planning a single-agent campus tour.

The tour must visit these locations in an optimal order:
- Library (open 9AM-5PM)
- Student Center (open 9AM-5PM)
- Engineering Building (open 9AM-5PM)
- Sports Complex (open 9AM-5PM)

Current time is 10:00 AM. You need to:
1. Check if each location is open at your planned arrival time
2. Visit each location in order
3. Complete the tour before 5PM

Use the available tools to plan and execute the campus tour.""",

    "P2": """You are coordinating multiple campus tour groups.

You have 3 tour guides (guide_1, guide_2, guide_3) and 4 visitor groups (group_A, group_B, group_C, group_D).
Each guide can handle at most 2 groups.

You need to:
1. Check each guide's availability
2. Assign guides to groups for tours starting at 10:00 AM, 11:00 AM, 1:00 PM, 2:00 PM
3. Ensure no guide is overbooked

Use the available tools to coordinate the tour assignments.""",

    "P3": """You are managing an urban ride-sharing service.

You have 3 vehicles (vehicle_1, vehicle_2, vehicle_3) with capacity of 4 passengers each.
5 passengers need rides:
- passenger_1: Downtown to Airport
- passenger_2: Airport to University
- passenger_3: University to Downtown
- passenger_4: Downtown to Mall
- passenger_5: Mall to Airport

You need to:
1. Check vehicle capacities
2. Book rides efficiently, grouping passengers where possible
3. Handle any capacity issues

Use the available tools to coordinate the rides.""",

    "P4": """You are managing ride-sharing with real-time disruptions.

Book rides for passengers with these routes:
- passenger_1: Route A (Downtown-Airport)
- passenger_2: Route B (Airport-University)
- passenger_3: Route C (may need route update)

During execution, Route C may be blocked. You need to:
1. Book initial rides
2. Check and update routes if disruptions occur
3. Handle blocked routes by finding alternatives

Use the available tools to manage the rides.""",

    "P5": """You are planning wedding logistics.

You need to coordinate:
1. Book a venue (grand_hall) for 200 guests
2. Book catering service (gourmet_catering) for 200 guests, premium menu
3. Book a live band (jazz_band) for 4 hours - audience size is 100 people

Complete all three bookings for the wedding event.

Use the available tools to make all arrangements.""",

    "P6": """You are planning a Thanksgiving dinner with family arrivals.

Tasks:
1. Schedule airport pickup for Uncle Bob arriving at 2:00 PM (driver: dad)
2. Schedule airport pickup for Aunt Mary arriving at 3:00 PM (driver: mom)
3. Assign cooking tasks:
   - Turkey preparation to grandma at 10:00 AM
   - Stuffing to mom at 1:00 PM
   - Dessert to sister at 2:00 PM
4. Check kitchen can handle 4 cooks at once

Coordinate all pickups and cooking assignments.

Use the available tools to plan the dinner.""",

    "P7": """You are coordinating disaster relief operations.

Deploy teams and supplies to affected regions:
1. Deploy medical team (team_medical) to Region A
2. Deploy rescue team (team_rescue) to Region B
3. Allocate 500 units of medical_supplies to Region A
4. Allocate 2000 units of food supplies to Region B (may exceed capacity)

Coordinate all deployments and handle any supply shortages.

Use the available tools to manage relief operations.""",

    "P8": """You are managing wedding day transportation with disruptions.

Book transport for the wedding party:
1. Book limo (limo_1) for bride party via hotel-church route (check route first)
2. Book van (van_1) for groomsmen via alternative route
3. Book venue for reception

Routes may be blocked due to road work. Check routes and book available transport.

Use the available tools to coordinate transportation.""",

    "P9": """You are managing Thanksgiving with flight disruptions.

Family arrivals:
1. Uncle Bob on flight1 (check status - may be delayed)
2. Aunt Mary on flight2 (on time)

You need to:
1. Check flight statuses
2. Schedule pickups (reschedule if flights are delayed)
3. Assign cooking tasks around the updated schedule

Coordinate arrivals and cooking schedule.

Use the available tools to manage the day.""",

    "P10": """You are managing GPU supply chain logistics.

Order components for GPU production:
1. Order 300 memory chips from supplier_1
2. Order 600 processors from supplier_2 (check capacity first - may exceed 500 limit)
3. Schedule assembly at facility_1 starting 2024-01-15

Coordinate orders and handle any capacity issues.

Use the available tools to manage the supply chain.""",

    "P11": """You are solving a job shop scheduling problem.

Schedule 3 jobs on 3 machines:
1. Job_1 on machine_1 (check availability first - may be down) - start: 0, duration: 10
2. Job_2 on machine_2 - start: 5, duration: 15
3. Job_3 on machine_3 - start: 10, duration: 20

Check machine availability and schedule jobs efficiently.

Use the available tools to create the schedule.""",
}


class CompensationLibRunner(BaseFrameworkRunner):
    """
    Framework runner using langchain-compensation for P1-P11 tasks.

    Single agent with compensation middleware vs SagaLLM's multi-agent approach.
    """

    def __init__(self):
        super().__init__()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0
        )
        logger.info("CompensationLibRunner initialized")

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        return self.run_task(task_definition)

    def run_task(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute a P1-P11 task using compensation-enabled agent."""
        start_time = time.time()
        self._record_memory_usage()

        task_id = task_definition.task_id
        reset_state()  # Clean state for each run

        # Get task-specific tools and prompt
        tools = TASK_TOOLS.get(task_id, ALL_TOOLS)
        prompt = TASK_PROMPTS.get(task_id, f"Execute planning task: {task_id}")

        # Create agent with compensation for this task
        agent = create_comp_agent(
            model=self.llm,
            tools=tools,
            compensation_mapping=COMPENSATION_MAPPING,
            system_prompt=f"""You are a planning agent for {task_definition.name}.

Complete the requested planning task using the available tools.
If any action fails, the system will automatically rollback previous successful actions.

Task Description: {task_definition.description}"""
        )

        logger.info(f"Running task {task_id}: {task_definition.name}")

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={
                    "run_name": f"compensation-{task_id}",
                    "tags": ["compensation-lib", task_id],
                    "metadata": {
                        "framework": "compensation_lib",
                        "task_id": task_id,
                        "task_name": task_definition.name,
                    }
                }
            )

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()

            return self._process_result(result, task_definition, execution_time)

        except Exception as e:
            logger.error(f"Task {task_id} error: {e}")
            import traceback
            traceback.print_exc()

            execution_time = time.time() - start_time
            return self._create_error_result(execution_time)

    def _process_result(self, result: Dict[str, Any], task_def: TaskDefinition, exec_time: float) -> Dict[str, Any]:
        """Process agent result and extract metrics."""
        messages = result.get("messages", [])

        # Extract tool calls and check for compensation
        tool_calls = []
        compensation_calls = []
        achieved_goals = []

        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_calls.append({
                        "tool": tool_name,
                        "args": tc.get("args", {}),
                    })

                    # Check if compensation tool was called
                    if tool_name in COMPENSATION_MAPPING.values():
                        compensation_calls.append(tool_name)

                    # Check if goal-related tool succeeded
                    if tool_name in COMPENSATION_MAPPING.keys():
                        achieved_goals.append(tool_name)

        # Extract LLM metrics
        llm_metrics = {"llm_call_count": 1, "total_input_tokens": 0, "total_output_tokens": 0}
        for msg in messages:
            if hasattr(msg, 'response_metadata'):
                usage = msg.response_metadata.get('usage_metadata', {})
                llm_metrics["total_input_tokens"] += usage.get('prompt_token_count', 0)
                llm_metrics["total_output_tokens"] += usage.get('candidates_token_count', 0)
        llm_metrics["total_tokens"] = llm_metrics["total_input_tokens"] + llm_metrics["total_output_tokens"]

        # Calculate goal satisfaction
        total_goals = len(task_def.goals)
        achieved = min(len(achieved_goals), total_goals)
        goal_satisfaction = achieved / total_goals if total_goals > 0 else 0

        exec_result = self._create_execution_result(
            achieved_goals=[g.goal_id for g in task_def.goals[:achieved]],
            satisfied_constraints=[c.constraint_id for c in task_def.constraints],
            schedule=tool_calls
        )

        exec_result['resource_usage']['llm_metrics'] = llm_metrics
        exec_result['resource_usage']['execution_time'] = exec_time
        exec_result['resource_usage']['compensation_metrics'] = {
            "rollback_triggered": len(compensation_calls) > 0,
            "actions_compensated": len(compensation_calls),
            "compensation_success": True if compensation_calls else False,
            "total_tool_calls": len(tool_calls),
        }
        exec_result['metrics'] = {
            "goal_satisfaction_rate": goal_satisfaction,
            "execution_time": exec_time,
        }

        return exec_result

    def _create_error_result(self, exec_time: float) -> Dict[str, Any]:
        """Create result for failed execution."""
        result = self._create_execution_result([], [], [])
        result['resource_usage']['llm_metrics'] = {
            "llm_call_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        }
        result['resource_usage']['execution_time'] = exec_time
        result['resource_usage']['compensation_metrics'] = {
            "rollback_triggered": False,
            "actions_compensated": 0,
            "compensation_success": False,
        }
        return result
