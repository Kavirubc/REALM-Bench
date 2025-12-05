"""
langchain-compensation router for REALM-Bench.

Uses the langchain-compensation library for automatic rollback on failure.
"""

import os
import sys
import time
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Add paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared_tools.tool_registry import get_tools_for_task, get_compensation_mapping_for_task
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

logger = get_logger("realm_bench.compensation")


class CompensationRouter:
    """
    Router using langchain-compensation library for automatic rollback.

    When any tool fails, the library automatically compensates all
    previously successful actions in reverse order.
    """

    def __init__(
        self,
        task_definition,
        model: str = "gemini-2.0-flash",
        log_file: Optional[str] = None
    ):
        """
        Initialize the compensation router.

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

        # Get tools and mappings for this task
        self.tools = get_tools_for_task(task_definition)
        self.compensation_mapping = get_compensation_mapping_for_task(task_definition)

        logger.info(
            f"CompensationRouter initialized for task {task_definition.task_id}"
        )
        logger.info(f"Tools: {[getattr(t, 'name', str(t)) for t in self.tools]}")
        logger.info(f"Compensation mapping: {self.compensation_mapping}")

        # Get system prompt for task category
        self.system_prompt = get_prompt_for_category(task_definition.category)

        # Initialize the model
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

        # Create the agent using langchain-compensation library
        # This provides automatic rollback when tools fail
        from langchain_compensation import create_comp_agent, ContentDictStrategy

        self.agent = create_comp_agent(
            model=llm,
            tools=self.tools,
            compensation_mapping=self.compensation_mapping,
            error_strategies=[ContentDictStrategy()],
            sequential_execution=True,  # Force sequential tool execution for Gemini
            debug=True,  # Enable debug logging
        ).with_config({
            "run_name": f"LangChain-Compensation-{task_definition.task_id}",
            "tags": ["langchain-compensation", "auto-rollback", task_definition.task_id],
        })

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the agent with automatic compensation support.

        Args:
            query: Task description/query for the agent.

        Returns:
            Dictionary with execution results including:
            - success: bool
            - messages: List of agent messages
            - final_state: ExecutionState snapshot
            - disruptions_triggered: List of triggered disruptions
            - action_history: List of actions performed
            - compensation_events: List of compensation events
        """
        # Reset state for fresh execution
        self.state_manager.reset()
        self.disruption_engine.reset()
        self.disruption_engine.configure_from_task(self.task_definition)

        log_task_start(logger, self.task_definition.task_id, "langchain_compensation")
        start_time = time.time()

        try:
            # Execute the agent - library handles compensation automatically
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })

            # Extract results
            final_state = self.state_manager.get_state()
            triggered_disruptions = self.disruption_engine.get_triggered_disruptions()

            duration = time.time() - start_time
            log_task_end(
                logger,
                self.task_definition.task_id,
                "langchain_compensation",
                True,
                duration
            )

            return {
                "success": True,
                "messages": result.get("messages", []),
                "final_state": final_state,
                "disruptions_triggered": triggered_disruptions,
                "action_history": final_state.action_history,
                "compensation_events": final_state.compensation_history,
                "execution_time": duration,
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Execution failed: {str(e)}")
            log_task_end(
                logger,
                self.task_definition.task_id,
                "langchain_compensation",
                False,
                duration
            )

            return {
                "success": False,
                "error": str(e),
                "final_state": self.state_manager.get_state(),
                "disruptions_triggered": self.disruption_engine.get_triggered_disruptions(),
                "action_history": self.state_manager.get_state().action_history,
                "compensation_events": self.state_manager.get_state().compensation_history,
                "execution_time": duration,
            }


def run_compensation_agent(query: str, task_definition=None) -> Dict[str, Any]:
    """
    Convenience function to run compensation agent.

    Args:
        query: Task description/query.
        task_definition: Optional TaskDefinition. If not provided,
            uses a default configuration.

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

    router = CompensationRouter(task_definition)
    return router.run(query)
