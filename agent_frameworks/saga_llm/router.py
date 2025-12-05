"""
SagaLLM router for REALM-Bench.

Implements the SAGA pattern for LLM agents with explicit rollback on failure.
Based on the SagaLLM framework approach but adapted for REALM-Bench tools.

Key differences from langchain-compensation:
- Uses ReAct loop with XML-based tool calling
- Explicit rollback() calls instead of automatic compensation
- Tracks executed actions and rolls back in reverse order on failure
"""

import os
import sys
import re
import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

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

logger = get_logger("realm_bench.saga_llm")


# ReAct system prompt with XML tool calling format (from SagaLLM)
REACT_SYSTEM_PROMPT = """You are a planning agent that executes tasks step-by-step using available tools.

For each step, think about what needs to be done, then call the appropriate tool.

## Tool Calling Format
When you need to call a tool, use the following XML format:
<tool_call>
{{"name": "<tool_name>", "arguments": {{"arg1": "value1", "arg2": "value2"}}, "id": <call_id>}}
</tool_call>

## Response Format
When you have completed the task or cannot proceed, provide a final response:
<response>
Your final response summarizing what was accomplished or why the task could not be completed.
</response>

## Thinking Format
Before each action, explain your reasoning:
<thought>
Your reasoning about what to do next.
</thought>

## Rules
1. Call tools ONE AT A TIME - wait for each result before proceeding
2. Each tool call must have a unique incrementing id starting from 1
3. After receiving a tool result, think about the next step
4. If a tool fails with an error, the system will automatically rollback previous actions
5. Continue until all goals are achieved or you encounter an unrecoverable error

## Available Tools
{tool_descriptions}
"""


@dataclass
class ExecutedAction:
    """Record of an executed action for rollback purposes."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    call_id: int


@dataclass
class SagaExecutionContext:
    """Context for SAGA execution tracking."""
    executed_actions: List[ExecutedAction] = field(default_factory=list)
    compensation_history: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    tool_call_counter: int = 0


class SagaLLMRouter:
    """
    Router implementing SAGA pattern for LLM agents.

    When any tool fails, explicitly rolls back all previously
    successful actions in reverse order using compensation tools.
    """

    def __init__(
        self,
        task_definition,
        model: str = "gemini-2.0-flash",
        log_file: Optional[str] = None,
        max_iterations: int = 20
    ):
        """
        Initialize the SAGA LLM router.

        Args:
            task_definition: TaskDefinition for the task to execute.
            model: LLM model to use (default: gemini-2.0-flash).
            log_file: Optional path to JSON log file.
            max_iterations: Maximum ReAct loop iterations.
        """
        self.task_definition = task_definition
        self.model_name = model
        self.max_iterations = max_iterations

        # Set up logging
        if log_file:
            setup_benchmark_logging(log_file)

        # Initialize state and disruption engine
        self.state_manager = StateManager()
        self.disruption_engine = DisruptionEngine()
        self.disruption_engine.configure_from_task(task_definition)

        # Get tools and mappings for this task
        self.langchain_tools = get_tools_for_task(task_definition)
        self.compensation_mapping = get_compensation_mapping_for_task(task_definition)

        # Create tool lookup by name
        self.tools_by_name: Dict[str, Callable] = {}
        for tool in self.langchain_tools:
            tool_name = getattr(tool, 'name', None)
            if tool_name:
                self.tools_by_name[tool_name] = tool

        logger.info(f"SagaLLMRouter initialized for task {task_definition.task_id}")
        logger.info(f"Tools: {list(self.tools_by_name.keys())}")
        logger.info(f"Compensation mapping: {self.compensation_mapping}")

        # Get system prompt for task category
        self.category_prompt = get_prompt_for_category(task_definition.category)

        # Initialize the LLM client
        self._init_llm(model)

    def _init_llm(self, model: str):
        """Initialize the LLM client based on model name."""
        if model.startswith("gemini"):
            # Use langchain_google_genai with REST transport (same as other routers)
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ.get('GOOGLE_API_KEY')
            self.llm_client = ChatGoogleGenerativeAI(
                model=model,
                temperature=0,
                google_api_key=api_key,
                transport='rest',  # Use REST API, not gRPC (avoids Vertex AI auth)
            )
            self.llm_type = "langchain"
        else:
            # Use langchain_openai for consistency
            from langchain_openai import ChatOpenAI
            self.llm_client = ChatOpenAI(model=model, temperature=0)
            self.llm_type = "langchain"

    def _get_tool_descriptions(self) -> str:
        """Generate tool descriptions for the system prompt."""
        descriptions = []
        for tool_name, tool in self.tools_by_name.items():
            # Get tool description from LangChain tool
            doc = getattr(tool, 'description', '') or tool.__doc__ or ''

            # Get argument schema
            args_schema = getattr(tool, 'args_schema', None)
            if args_schema:
                schema_dict = args_schema.model_json_schema()
                properties = schema_dict.get('properties', {})
                required = schema_dict.get('required', [])

                args_desc = []
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get('type', 'any')
                    prop_desc = prop_info.get('description', '')
                    req_marker = '*' if prop_name in required else ''
                    args_desc.append(f"    - {prop_name}{req_marker} ({prop_type}): {prop_desc}")

                args_str = "\n".join(args_desc) if args_desc else "    (no arguments)"
            else:
                args_str = "    (see function signature)"

            descriptions.append(f"### {tool_name}\n{doc}\nArguments:\n{args_str}")

        return "\n\n".join(descriptions)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with messages and return response text."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        # Convert to LangChain message format
        langchain_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        # Call the LLM (works for both Gemini and OpenAI via LangChain)
        response = self.llm_client.invoke(langchain_messages)
        return response.content

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from XML tags in response."""
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON: {e}")
                return None
        return None

    def _parse_response(self, text: str) -> Optional[str]:
        """Extract final response from XML tags."""
        pattern = r'<response>\s*(.*?)\s*</response>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given arguments."""
        if tool_name not in self.tools_by_name:
            return {"status": "error", "error": f"Unknown tool: {tool_name}"}

        tool = self.tools_by_name[tool_name]
        try:
            # LangChain tools are invoked differently
            result = tool.invoke(arguments)

            # Ensure result is a dict
            if isinstance(result, str):
                result = {"status": "success", "message": result}
            elif not isinstance(result, dict):
                result = {"status": "success", "result": str(result)}

            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _rollback(self, context: SagaExecutionContext) -> List[Dict[str, Any]]:
        """
        Roll back executed actions - ORIGINAL SAGALLM BEHAVIOR.

        The original SagaLLM library's default rollback() just prints a message
        and does NOT actually execute compensation tools. This simulates that
        behavior to demonstrate the difference vs langchain-compensation.

        See: SagaLLM/src/multi_agent/agent.py lines 227-229
            def rollback(self):
                print(f"Rolling back {self.name}'s operation...")  # JUST A PRINT!
        """
        rollback_results = []

        # ORIGINAL SAGALLM BEHAVIOR: Just print, don't actually compensate!
        for action in reversed(context.executed_actions):
            # This is what the original SagaLLM does - just logs/prints
            logger.info(f"[SagaLLM Default] Rolling back {action.tool_name}'s operation...")
            print(f"Rolling back {action.tool_name}'s operation...")  # Original behavior

            # Record that we "attempted" rollback but didn't actually do anything
            rollback_results.append({
                "original_action": action.tool_name,
                "compensation_tool": None,
                "status": "no_op",
                "reason": "Original SagaLLM default rollback() is a no-op print statement"
            })

        context.compensation_history.extend(rollback_results)
        return rollback_results

    def _build_compensation_args(self, action: ExecutedAction) -> Dict[str, Any]:
        """Build arguments for compensation tool from original action."""
        # Different tools need different compensation argument mappings
        tool_name = action.tool_name
        original_args = action.arguments
        original_result = action.result

        if tool_name == "schedule_job":
            return {"job_id": original_args.get("job_id")}
        elif tool_name == "assign_vehicle":
            return {
                "vehicle_id": original_args.get("vehicle_id"),
                "route_id": original_args.get("route_id")
            }
        elif tool_name == "allocate_resource":
            return {
                "resource_id": original_args.get("resource_id"),
                "task_id": original_args.get("task_id")
            }
        elif tool_name == "assign_task":
            return {
                "task_id": original_args.get("task_id"),
                "worker_id": original_args.get("worker_id")
            }
        elif tool_name == "deploy_team":
            return {
                "team_id": original_args.get("team_id"),
                "location_id": original_args.get("location_id")
            }
        elif tool_name == "allocate_supplies":
            return {
                "supply_id": original_args.get("supply_id"),
                "location_id": original_args.get("location_id"),
                "quantity": original_args.get("quantity")
            }
        else:
            # Default: pass all original arguments
            return original_args

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the agent with SAGA-style rollback on failure.

        Args:
            query: Task description/query for the agent.

        Returns:
            Dictionary with execution results.
        """
        # Reset state for fresh execution
        self.state_manager.reset()
        self.disruption_engine.reset()
        self.disruption_engine.configure_from_task(self.task_definition)

        log_task_start(logger, self.task_definition.task_id, "saga_llm")
        start_time = time.time()

        # Set up LangSmith tracing context
        from langsmith import traceable
        return self._run_with_tracing(query, start_time)

    @property
    def _trace_name(self) -> str:
        return f"SagaLLM-{self.task_definition.task_id}"

    def _run_with_tracing(self, query: str, start_time: float) -> Dict[str, Any]:
        """Run with LangSmith tracing wrapper."""
        try:
            from langsmith.run_helpers import traceable

            @traceable(
                name=self._trace_name,
                tags=["saga-llm", "explicit-rollback", self.task_definition.task_id],
            )
            def traced_run():
                return self._execute_react_loop(query, start_time)

            return traced_run()
        except ImportError:
            # LangSmith not available, run without tracing
            return self._execute_react_loop(query, start_time)

    def _execute_react_loop(self, query: str, start_time: float) -> Dict[str, Any]:
        """Execute the ReAct loop (extracted for tracing)."""

        # Initialize execution context
        context = SagaExecutionContext()

        # Build system prompt with tool descriptions
        tool_descriptions = self._get_tool_descriptions()
        system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

        # Add category-specific guidance
        full_system = f"{system_prompt}\n\n## Task-Specific Guidance\n{self.category_prompt}"

        # Initialize message history
        context.messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": query}
        ]

        try:
            # ReAct loop
            for iteration in range(self.max_iterations):
                logger.info(f"ReAct iteration {iteration + 1}")

                # Get LLM response
                response_text = self._call_llm(context.messages)
                logger.debug(f"LLM response: {response_text[:500]}...")

                # Add assistant response to history
                context.messages.append({"role": "assistant", "content": response_text})

                # Check for final response
                final_response = self._parse_response(response_text)
                if final_response:
                    logger.info(f"Agent completed with response: {final_response[:200]}...")
                    break

                # Check for tool call
                tool_call = self._parse_tool_call(response_text)
                if tool_call:
                    tool_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})
                    call_id = tool_call.get("id", context.tool_call_counter + 1)
                    context.tool_call_counter = call_id

                    logger.info(f"Executing tool: {tool_name} with args: {arguments}")

                    # Execute the tool
                    result = self._execute_tool(tool_name, arguments)
                    logger.info(f"Tool result: {result}")

                    # Check if tool failed
                    if result.get("status") == "error":
                        logger.warning(f"Tool {tool_name} failed: {result.get('error')}")

                        # SAGA ROLLBACK: Roll back all previous actions
                        logger.info("Initiating SAGA rollback...")
                        rollback_results = self._rollback(context)

                        # Add failure and rollback info to messages
                        rollback_msg = f"Tool {tool_name} failed: {result.get('error')}\n"
                        rollback_msg += f"SAGA rollback executed for {len(context.executed_actions)} previous actions.\n"
                        rollback_msg += f"Rollback results: {json.dumps(rollback_results, indent=2)}"

                        context.messages.append({
                            "role": "user",
                            "content": f"Tool result (call_id={call_id}):\n{rollback_msg}"
                        })

                        # Clear executed actions since they've been rolled back
                        context.executed_actions = []
                    else:
                        # Record successful action for potential rollback
                        action = ExecutedAction(
                            tool_name=tool_name,
                            arguments=arguments,
                            result=result,
                            call_id=call_id
                        )
                        context.executed_actions.append(action)

                        # Add result to messages
                        context.messages.append({
                            "role": "user",
                            "content": f"Tool result (call_id={call_id}):\n{json.dumps(result, indent=2)}"
                        })
                else:
                    # No tool call or response - prompt to continue
                    logger.warning("No tool call or response found, prompting agent...")
                    context.messages.append({
                        "role": "user",
                        "content": "Please continue with the next action or provide a <response> if the task is complete."
                    })

            # Extract final state
            final_state = self.state_manager.get_state()
            triggered_disruptions = self.disruption_engine.get_triggered_disruptions()

            duration = time.time() - start_time
            log_task_end(
                logger,
                self.task_definition.task_id,
                "saga_llm",
                True,
                duration
            )

            return {
                "success": True,
                "messages": context.messages,
                "final_state": final_state,
                "disruptions_triggered": triggered_disruptions,
                "action_history": final_state.action_history,
                "compensation_events": context.compensation_history,
                "executed_actions": [
                    {"tool": a.tool_name, "args": a.arguments, "result": a.result}
                    for a in context.executed_actions
                ],
                "execution_time": duration,
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Execution failed: {str(e)}")

            # Attempt rollback on exception
            logger.info("Exception occurred, attempting SAGA rollback...")
            try:
                self._rollback(context)
            except Exception as rollback_error:
                logger.error(f"Rollback also failed: {rollback_error}")

            log_task_end(
                logger,
                self.task_definition.task_id,
                "saga_llm",
                False,
                duration
            )

            return {
                "success": False,
                "error": str(e),
                "final_state": self.state_manager.get_state(),
                "disruptions_triggered": self.disruption_engine.get_triggered_disruptions(),
                "action_history": self.state_manager.get_state().action_history,
                "compensation_events": context.compensation_history,
                "execution_time": duration,
            }


def run_saga_agent(query: str, task_definition=None) -> Dict[str, Any]:
    """
    Convenience function to run SAGA LLM agent.

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

    router = SagaLLMRouter(task_definition)
    return router.run(query)
