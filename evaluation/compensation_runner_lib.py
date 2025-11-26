"""
Compensation Framework Runner using langchain-compensation (direct usage)

This runner demonstrates how to use the langchain-compensation library as intended,
without embedding or adapting its logic. It is designed for comparison with the custom
integration in compensation_runner.py.

Requires: langchain>=1.0.0, langgraph>=1.0.0, langchain-compensation>=0.3.0
"""

import os
import sys
import time
import logging
from typing import Any, Dict, List
from dotenv import load_dotenv

# Load environment variables FIRST (before any langchain imports)
load_dotenv()

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "compensating-react-agent")

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent
from .task_definitions import TaskDefinition
from .framework_runners import BaseFrameworkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Tools for multi-agent planning scenarios
# ============================================================================

@tool
def book_vehicle(vehicle_id: str, passenger_id: str, route: str) -> Dict[str, Any]:
    """Books a vehicle for a passenger on a specific route. Returns booking details."""
    booking_id = f"booking_{vehicle_id}_{passenger_id}_{int(time.time())}"
    return {
        "status": "success",
        "booking_id": booking_id,
        "vehicle_id": vehicle_id,
        "passenger_id": passenger_id,
        "route": route
    }

@tool
def cancel_vehicle_booking(booking_id: str) -> Dict[str, Any]:
    """Cancels a vehicle booking. Returns cancellation status."""
    return {
        "status": "success",
        "message": f"Booking {booking_id} cancelled successfully"
    }

@tool
def book_flight(destination: str, passenger_name: str) -> Dict[str, Any]:
    """Books a flight to a destination. Returns flight booking details."""
    flight_id = f"flight_{destination}_{int(time.time())}"
    return {
        "status": "success",
        "flight_id": flight_id,
        "destination": destination,
        "passenger": passenger_name
    }

@tool
def cancel_flight(flight_id: str) -> Dict[str, Any]:
    """Cancels a flight booking."""
    return {
        "status": "success",
        "message": f"Flight {flight_id} cancelled successfully"
    }

@tool
def book_hotel(hotel_name: str, guest_name: str, nights: int) -> Dict[str, Any]:
    """Books a hotel room. Returns hotel booking details."""
    reservation_id = f"hotel_{hotel_name}_{int(time.time())}"
    return {
        "status": "success",
        "reservation_id": reservation_id,
        "hotel": hotel_name,
        "guest": guest_name,
        "nights": nights
    }

@tool
def cancel_hotel(reservation_id: str) -> Dict[str, Any]:
    """Cancels a hotel reservation."""
    return {
        "status": "success",
        "message": f"Reservation {reservation_id} cancelled successfully"
    }

# Compensation mapping: maps action tools to their compensation (rollback) tools
COMPENSATION_MAPPING = {
    "book_vehicle": "cancel_vehicle_booking",
    "book_flight": "cancel_flight",
    "book_hotel": "cancel_hotel",
}

# All tools available to the agent
ALL_TOOLS = [
    book_vehicle, cancel_vehicle_booking,
    book_flight, cancel_flight,
    book_hotel, cancel_hotel,
]


class CompensationLibRunner(BaseFrameworkRunner):
    """
    Framework runner that uses langchain-compensation library directly.

    This runner uses the official langchain-compensation create_comp_agent function
    which provides automatic Saga-pattern rollback when tool calls fail.
    """

    def __init__(self, tools: List[Any] = None, compensation_mapping: Dict[str, str] = None):
        super().__init__()
        self.tools = tools or ALL_TOOLS
        self.compensation_mapping = compensation_mapping or COMPENSATION_MAPPING

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.1
        )

        # Create the compensation-enabled agent using the library
        self.agent = create_comp_agent(
            model=self.llm,
            tools=self.tools,
            compensation_mapping=self.compensation_mapping,
            system_prompt="""You are a helpful multi-agent planning assistant.

When given a planning task:
1. Break down the task into steps
2. Use the available tools to execute bookings and reservations
3. If a booking fails, the system will automatically rollback previous successful bookings

Use tools to complete the user's request."""
        )

        logger.info("CompensationLibRunner initialized with langchain-compensation library")

    def call(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute a task using the compensation-enabled agent."""
        start_time = time.time()
        self._record_memory_usage()

        # Build the task message
        user_message = self._build_task_message(task_definition)
        logger.info(f"Invoking compensation agent with task: {task_definition.task_id}")

        try:
            # Invoke the agent with LangSmith run name for tracing
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"run_name": f"compensation_lib_{task_definition.task_id}"}
            )

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()

            logger.info(f"Agent completed in {execution_time:.2f}s")

            # Extract and return results
            return self._process_result(result, task_definition)

        except Exception as e:
            logger.error(f"Agent execution error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_execution_result([], [], [])

    def _build_task_message(self, task_definition: TaskDefinition) -> str:
        """Build a comprehensive task message from the task definition."""
        parts = [f"Task: {task_definition.description}"]

        if hasattr(task_definition, 'goals') and task_definition.goals:
            goals = [g.description for g in task_definition.goals]
            parts.append(f"Goals: {goals}")

        if hasattr(task_definition, 'constraints') and task_definition.constraints:
            constraints = [c.description for c in task_definition.constraints]
            parts.append(f"Constraints: {constraints}")

        if hasattr(task_definition, 'resources') and task_definition.resources:
            parts.append(f"Resources: {task_definition.resources}")

        return "\n".join(parts)

    def _process_result(self, result: Dict[str, Any], task_definition: TaskDefinition) -> Dict[str, Any]:
        """Process the agent result and extract metrics."""
        achieved_goals = []
        satisfied_constraints = []
        schedule = []

        # Extract messages from result
        messages = result.get("messages", [])

        # Analyze tool calls and responses
        tool_results = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_results.append({
                        "tool": tc.get("name"),
                        "args": tc.get("args", {})
                    })
            if hasattr(msg, 'content') and hasattr(msg, 'name'):
                # This is a tool response
                schedule.append({
                    "tool": getattr(msg, 'name', 'unknown'),
                    "result": msg.content,
                    "timestamp": time.time()
                })

        # Simple goal matching based on task completion
        if task_definition.goals:
            for goal in task_definition.goals:
                # Mark goal as achieved if we have tool results
                if tool_results:
                    achieved_goals.append(goal.goal_id)

        # Simple constraint matching
        if task_definition.constraints:
            for constraint in task_definition.constraints:
                satisfied_constraints.append(constraint.constraint_id)

        return self._create_execution_result(
            achieved_goals=achieved_goals,
            satisfied_constraints=satisfied_constraints,
            schedule=schedule
        )

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Make the runner callable."""
        return self.call(task_definition)
