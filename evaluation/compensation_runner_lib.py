"""
Compensation Framework Runner using langchain-compensation library.

This runner uses the langchain-compensation library which provides:
- create_comp_agent: Creates a LangChain agent with automatic compensation middleware
- CompensationMiddleware: Wraps tool calls and handles automatic LIFO/DAG rollback
- CompensationLog: Thread-safe tracking of actions and dependencies

This is benchmarked against SagaLLM which uses a multi-agent Saga pattern.
"""

import os
import sys
import time
import logging
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "c-benchmark")

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent
from .task_definitions import TaskDefinition
from .framework_runners import BaseFrameworkRunner

# Import the SAME tools used by SagaLLM
from .compensation_runner import (
    book_vehicle, cancel_vehicle_booking,
    allocate_resource, deallocate_resource,
    check_capacity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compensation mapping: tool -> compensation tool
COMPENSATION_MAPPING = {
    "book_vehicle": "cancel_vehicle_booking",
    "allocate_resource": "deallocate_resource",
}

# All tools for the agent
ALL_TOOLS = [
    book_vehicle, cancel_vehicle_booking,
    allocate_resource, deallocate_resource,
    check_capacity,
]


class CompensationLibRunner(BaseFrameworkRunner):
    """
    Framework runner using langchain-compensation's create_comp_agent.

    Key difference from SagaLLM:
    - Single agent with compensation middleware
    - LLM decides which tools to call
    - Middleware automatically tracks and rolls back on failure
    """

    def __init__(self):
        super().__init__()

        # Initialize LLM (same as SagaLLM uses)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0
        )

        # Create agent with compensation middleware using the library
        self.agent = create_comp_agent(
            model=self.llm,
            tools=ALL_TOOLS,
            compensation_mapping=COMPENSATION_MAPPING,
            system_prompt="""You are a task execution agent with automatic rollback capabilities.

Execute the tasks step by step using the available tools:
- book_vehicle(vehicle_id, passenger_id, route): Book a vehicle/venue
- allocate_resource(resource_type, resource_id, amount): Allocate a resource
- check_capacity(resource_type, requested_amount): Check if capacity is available (fails if amount > 50)

If any step fails, the system automatically rolls back previous operations.
Execute tasks in the exact order specified. Do not skip steps."""
        )

        logger.info("CompensationLibRunner initialized with langchain-compensation library")

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        return self.call(task_definition)

    def call(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute a task using the compensation-enabled agent."""
        start_time = time.time()
        self._record_memory_usage()

        task_id = task_definition.task_id
        user_message = self._build_task_message(task_id)

        logger.info(f"Invoking compensation agent with task: {task_id}")

        try:
            # Invoke the agent - it will use LLM to decide tool calls
            # CompensationMiddleware will track and rollback on failure
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"run_name": f"compensation_lib_{task_id}"}
            )

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()

            logger.info(f"Agent completed in {execution_time:.2f}s")

            # Extract results
            return self._process_result(result)

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_execution_result([], [], [])

    def _build_task_message(self, task_id: str) -> str:
        """Build task-specific message matching SagaLLM's tasks."""

        if task_id == "P5-ACID":
            return """Execute this wedding planning task with exactly 3 steps in order:

Step 1: Book venue using book_vehicle with:
- vehicle_id: "venue_1"
- passenger_id: "wedding_party"
- route: "main"

Step 2: Book caterer using allocate_resource with:
- resource_type: "caterer"
- resource_id: "cat_1"
- amount: 10

Step 3: Check band capacity using check_capacity with:
- resource_type: "band"
- requested_amount: 100

Note: Step 3 will fail because requested_amount > 50.
The system should automatically rollback steps 1 and 2."""

        elif task_id == "P6-ACID":
            return """Execute this Thanksgiving dinner ordering task with exactly 3 steps in order:

Step 1: Order sides using book_vehicle with:
- vehicle_id: "sides_truck"
- passenger_id: "food"
- route: "kitchen"

Step 2: Order drinks using book_vehicle with:
- vehicle_id: "drinks_truck"
- passenger_id: "beverages"
- route: "bar"

Step 3: Check turkey capacity using check_capacity with:
- resource_type: "turkey"
- requested_amount: 100

Note: Step 3 will fail because requested_amount > 50.
The system should automatically rollback steps 1 and 2."""

        else:
            return f"Execute task: {task_id}"

    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent result."""
        achieved_goals = []
        satisfied_constraints = []
        schedule = []

        # Extract messages from result
        messages = result.get("messages", [])

        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    schedule.append({
                        "tool": tc.get("name"),
                        "args": tc.get("args", {}),
                        "timestamp": time.time()
                    })

        return self._create_execution_result(
            achieved_goals=achieved_goals,
            satisfied_constraints=satisfied_constraints,
            schedule=schedule
        )
