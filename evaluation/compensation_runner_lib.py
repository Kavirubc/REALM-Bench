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
os.environ.setdefault("LANGSMITH_PROJECT", "realm-bench-compensation-lib")

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

# Import MongoDB tools for database workflows
try:
    from .mongodb_tools import (
        create_user, delete_user,
        update_user_profile, revert_user_profile,
        add_user_preferences, remove_user_preferences,
        create_user_session, delete_user_session,
        get_user_info
    )
    MONGODB_TOOLS_AVAILABLE = True
except ImportError:
    MONGODB_TOOLS_AVAILABLE = False
    logger.warning("MongoDB tools not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compensation mapping: tool -> compensation tool
COMPENSATION_MAPPING = {
    "book_vehicle": "cancel_vehicle_booking",
    "allocate_resource": "deallocate_resource",
}

# MongoDB compensation mapping
MONGODB_COMPENSATION_MAPPING = {
    "create_user": "delete_user",
    "update_user_profile": "revert_user_profile",
    "add_user_preferences": "remove_user_preferences",
    "create_user_session": "delete_user_session",
}

# All tools for the agent
ALL_TOOLS = [
    book_vehicle, cancel_vehicle_booking,
    allocate_resource, deallocate_resource,
    check_capacity,
]

# Add MongoDB tools if available
if MONGODB_TOOLS_AVAILABLE:
    ALL_TOOLS.extend([
        create_user, delete_user,
        update_user_profile, revert_user_profile,
        add_user_preferences, remove_user_preferences,
        create_user_session, delete_user_session,
        get_user_info
    ])
    COMPENSATION_MAPPING.update(MONGODB_COMPENSATION_MAPPING)


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
        # Note: The LLM should NOT know about compensation - it just tries to complete tasks
        # Compensation happens automatically in the background
        self.agent = create_comp_agent(
            model=self.llm,
            tools=ALL_TOOLS,
            compensation_mapping=COMPENSATION_MAPPING,
            system_prompt="""You are a helpful assistant that helps coordinate events and manage resources.

You have access to tools that can:
- book_vehicle(vehicle_id, passenger_id, route): Book vehicles or venues for transportation or events
- allocate_resource(resource_type, resource_id, amount): Allocate resources like catering services
- check_capacity(resource_type, requested_amount): Check if a resource has enough capacity available

When given a task, use the appropriate tools to complete it. Read the tool descriptions carefully to understand what parameters they need."""
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
                config={
                    "run_name": f"compensation-lib-{task_id}",
                    "tags": ["compensation-lib", "single-agent", task_id, "mongodb-workflow"],
                    "metadata": {
                        "framework": "compensation_lib",
                        "task_id": task_id,
                        "task_name": task_definition.name,
                        "agent_type": "single-agent",
                        "workflow": "mongodb-user-profile"
                    }
                }
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
        """Build natural task descriptions without hardcoded parameters or failure hints."""

        if task_id == "P5-ACID":
            return """You are planning a wedding event. You need to coordinate the following:

1. Book a venue for the wedding ceremony and reception
2. Arrange catering services for 200 guests
3. Book a live band for entertainment - they need to accommodate a large audience of 100 people

Please complete all three bookings. Use the available tools to make the necessary arrangements."""

        elif task_id == "P6-ACID":
            return """You are organizing a large Thanksgiving dinner for your extended family. You need to:

1. Order side dishes for the meal
2. Order drinks and beverages
3. Order a large turkey - you need enough for 100 people

Please place all the orders using the available tools."""

        elif task_id == "MONGODB-ACID":
            # Get user_id from task resources
            task_def = None
            try:
                from .task_definitions import TASK_DEFINITIONS
                from .compensation_tasks import COMPENSATION_TASK_DEFINITIONS
                all_tasks = {**TASK_DEFINITIONS, **COMPENSATION_TASK_DEFINITIONS}
                task_def = all_tasks.get(task_id)
            except:
                pass
            
            user_id = "test_user_123"  # Default
            if task_def and task_def.resources:
                user_id = task_def.resources.get("user_id", user_id)
            
            return f"""You need to set up a complete user profile in the database for user ID: {user_id}. This involves:

1. Creating a new user account with user_id: {user_id}
2. Updating the user's profile with additional information
3. Adding user preferences for the application
4. Creating an active session for the user

Please complete all steps to fully set up the user profile. Use the available database tools to perform these operations. Make sure to use the user_id: {user_id} for all operations."""

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
