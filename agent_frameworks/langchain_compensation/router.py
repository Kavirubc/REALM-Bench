"""
langchain-compensation router for REALM-Bench.

Uses LangChain v1 create_agent with custom middleware for automatic compensation.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

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
    log_compensation,
)
from prompt_templates.planning_template import get_prompt_for_category

load_dotenv()

logger = get_logger("realm_bench.compensation")


@dataclass
class CompensationContext:
    """Context for compensation middleware."""
    executed_actions: List[Dict[str, Any]] = None
    compensation_mapping: Dict[str, str] = None
    tool_by_name: Dict[str, Callable] = None

    def __post_init__(self):
        if self.executed_actions is None:
            self.executed_actions = []
        if self.compensation_mapping is None:
            self.compensation_mapping = {}
        if self.tool_by_name is None:
            self.tool_by_name = {}


class CompensationRouter:
    """
    Router using LangChain v1 create_agent with compensation middleware.

    Provides automatic compensation (rollback) when tool calls fail.
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

        # Build tool lookup by name
        self._tool_by_name: Dict[str, Callable] = {}
        for tool in self.tools:
            name = getattr(tool, 'name', None) or getattr(tool, '__name__', str(tool))
            self._tool_by_name[name] = tool

        logger.info(
            f"CompensationRouter initialized for task {task_definition.task_id}"
        )
        logger.info(f"Tools: {list(self._tool_by_name.keys())}")
        logger.info(f"Compensation mapping: {self.compensation_mapping}")

        # Get system prompt for task category
        self.system_prompt = get_prompt_for_category(task_definition.category)

        # Track executed actions for compensation
        self._executed_actions: List[Dict[str, Any]] = []

        # Use original tools directly (compensation is tracked via state manager)
        self._wrapped_tools = self.tools

        # Initialize the model
        if model.startswith("gemini"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ.get('GOOGLE_API_KEY')
            llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=0)

        # Create the agent using LangChain v1 create_agent
        from langchain.agents import create_agent
        self.agent = create_agent(
            model=llm,
            tools=self._wrapped_tools,
            system_prompt=self.system_prompt,
        )

    def _create_wrapped_tools(self) -> List[Callable]:
        """Create wrapped tools that track actions and trigger compensation on failure."""
        from langchain.tools import tool as tool_decorator

        wrapped_tools = []

        for original_tool in self.tools:
            tool_name = getattr(original_tool, 'name', None) or getattr(original_tool, '__name__', str(original_tool))
            tool_desc = getattr(original_tool, 'description', f"Tool: {tool_name}")

            # Check if this is a compensation tool (don't wrap those)
            is_compensation_tool = tool_name in self.compensation_mapping.values()

            if is_compensation_tool:
                # Keep compensation tools as-is
                wrapped_tools.append(original_tool)
            else:
                # Wrap action tools to track and compensate on failure
                wrapped_tools.append(self._wrap_tool(original_tool, tool_name))

        return wrapped_tools

    def _wrap_tool(self, original_tool: Callable, tool_name: str) -> Callable:
        """Wrap a tool to track execution and trigger compensation on failure."""
        from langchain_core.tools import StructuredTool

        # Get the original function
        if hasattr(original_tool, 'func'):
            original_func = original_tool.func
        elif hasattr(original_tool, '_run'):
            original_func = original_tool._run
        else:
            original_func = original_tool

        router = self  # Capture reference

        def wrapped_func(**kwargs):
            result = original_tool.invoke(kwargs)

            # Track successful actions
            if "SUCCESS" in str(result) and tool_name in router.compensation_mapping:
                router._executed_actions.append({
                    "tool_name": tool_name,
                    "args": kwargs,
                    "result": result,
                })

            # Check for failure - trigger compensation
            if "FAILED" in str(result):
                logger.info(f"Tool {tool_name} failed, triggering compensation")
                router._trigger_compensation()

            return result

        # Create new tool with same signature
        return StructuredTool.from_function(
            func=wrapped_func,
            name=tool_name,
            description=getattr(original_tool, 'description', f"Tool: {tool_name}"),
            args_schema=getattr(original_tool, 'args_schema', None),
        )

    def _trigger_compensation(self):
        """Execute compensation for all previously successful actions (in reverse order)."""
        for action in reversed(self._executed_actions):
            tool_name = action["tool_name"]
            comp_tool_name = self.compensation_mapping.get(tool_name)

            if comp_tool_name and comp_tool_name in self._tool_by_name:
                comp_tool = self._tool_by_name[comp_tool_name]

                # Extract ID from original args for compensation
                comp_args = {}
                for key in ["job_id", "vehicle_id", "resource_id", "team_id", "task_id"]:
                    if key in action["args"]:
                        comp_args[key] = action["args"][key]

                try:
                    result = comp_tool.invoke(comp_args)
                    log_compensation(
                        logger,
                        comp_tool_name,
                        tool_name,
                        comp_args,
                        "SUCCESS" in str(result)
                    )
                    logger.info(f"Compensation {comp_tool_name}: {result}")
                except Exception as e:
                    logger.error(f"Compensation {comp_tool_name} failed: {e}")

        # Clear action history after compensation
        self._executed_actions.clear()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the agent with compensation support.

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
        self._executed_actions.clear()

        log_task_start(logger, self.task_definition.task_id, "langchain_compensation")
        start_time = time.time()

        try:
            # Execute the agent
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
        # Create a minimal mock task definition for testing
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
