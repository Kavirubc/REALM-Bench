"""
LangGraph router V2 using shared tools for fair comparison.

Uses langchain.agents.create_agent for agent creation.
"""

import os
import sys
import time
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Add paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared_tools.tool_registry import get_tools_for_task
from shared_tools.state_manager import StateManager
from shared_tools.disruption_engine import DisruptionEngine
from shared_tools.logging_config import (
    setup_benchmark_logging,
    get_logger,
    log_task_start,
    log_task_end,
)
from prompt_templates.planning_template import get_prompt_for_category

load_dotenv()

logger = get_logger("realm_bench.langgraph")


class LangGraphRouterV2:
    """
    LangGraph router with shared tools (no automatic compensation).

    Uses langchain.agents.create_agent with a pre-initialized LLM.
    """

    def __init__(
        self,
        task_definition,
        model: str = "gemini-2.0-flash",
        log_file: Optional[str] = None
    ):
        """
        Initialize the LangGraph V2 router.

        Args:
            task_definition: TaskDefinition for the task to execute.
            model: LLM model to use (default: gemini-2.0-flash).
            log_file: Optional path to JSON log file.
        """
        self.task_definition = task_definition
        self.model_name = model

        # Set up logging
        if log_file:
            setup_benchmark_logging(log_file)

        # Initialize state and disruption engine
        self.state_manager = StateManager()
        self.disruption_engine = DisruptionEngine()
        self.disruption_engine.configure_from_task(task_definition)

        # Get tools for this task (same as compensation router)
        self.tools = get_tools_for_task(task_definition)

        logger.info(
            f"LangGraphRouterV2 initialized for task {task_definition.task_id}"
        )
        logger.info(f"Tools: {[getattr(t, 'name', str(t)) for t in self.tools]}")

        # Get system prompt for task category
        self.system_prompt = get_prompt_for_category(task_definition.category)

        # Initialize the model using langchain_google_genai (not Vertex)
        if model.startswith("gemini"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ.get('GOOGLE_API_KEY')
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0,
                google_api_key=api_key,
                transport='rest',  # Use REST API, not gRPC (avoids Vertex AI auth)
            )
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=0)

        # Create the agent using langchain.agents.create_agent
        from langchain.agents import create_agent
        self.agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        ).with_config({"recursion_limit": 50})  # Limit tool calls to prevent infinite loops

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the agent without automatic compensation.

        Args:
            query: Task description/query for the agent.

        Returns:
            Dictionary with execution results including:
            - success: bool
            - messages: List of agent messages
            - final_state: ExecutionState snapshot
            - disruptions_triggered: List of triggered disruptions
            - action_history: List of actions performed
            - compensation_events: Empty (no auto-compensation)
        """
        # Reset state for fresh execution
        self.state_manager.reset()
        self.disruption_engine.reset()
        self.disruption_engine.configure_from_task(self.task_definition)  # Re-configure after reset

        log_task_start(logger, self.task_definition.task_id, "langgraph_v2")
        start_time = time.time()

        try:
            # Execute the agent
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })

            # Extract results
            execution_state = self.state_manager.get_state()
            triggered_disruptions = self.disruption_engine.get_triggered_disruptions()

            duration = time.time() - start_time
            log_task_end(
                logger,
                self.task_definition.task_id,
                "langgraph_v2",
                True,
                duration
            )

            return {
                "success": True,
                "messages": result.get("messages", []),
                "final_state": execution_state,
                "disruptions_triggered": triggered_disruptions,
                "action_history": execution_state.action_history,
                "compensation_events": [],  # LangGraph has no auto-compensation
                "execution_time": duration,
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Execution failed: {str(e)}")
            log_task_end(
                logger,
                self.task_definition.task_id,
                "langgraph_v2",
                False,
                duration
            )

            return {
                "success": False,
                "error": str(e),
                "final_state": self.state_manager.get_state(),
                "disruptions_triggered": self.disruption_engine.get_triggered_disruptions(),
                "action_history": self.state_manager.get_state().action_history,
                "compensation_events": [],
                "execution_time": duration,
            }


def run_langgraph_v2_agent(query: str, task_definition=None) -> Dict[str, Any]:
    """
    Convenience function to run LangGraph V2 agent.

    Args:
        query: Task description/query.
        task_definition: Optional TaskDefinition.

    Returns:
        Execution result dictionary.
    """
    if task_definition is None:
        from dataclasses import dataclass
        from evaluation.task_definitions import TaskCategory

        @dataclass
        class MockTaskDefinition:
            task_id: str = "test"
            category: TaskCategory = TaskCategory.SCHEDULING
            disruption_scenarios: list = None

            def __post_init__(self):
                if self.disruption_scenarios is None:
                    self.disruption_scenarios = []

        task_definition = MockTaskDefinition()

    router = LangGraphRouterV2(task_definition)
    return router.run(query)
