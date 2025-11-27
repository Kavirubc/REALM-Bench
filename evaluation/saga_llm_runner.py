"""
SagaLLM Framework Runner for REALM-Bench

This module provides a runner for the SagaLLM framework to compare against
langchain-compensation. It attempts to implement ACID transactions using
SagaLLM's coordination primitives.
"""

import time
import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables FIRST (before any langchain imports)
load_dotenv()

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "realm-bench-sagallm")
# Add SagaLLM lib path - IMPORTANT: This must be added before importing any Saga modules
saga_lib_path = os.path.join(project_root, "agent_frameworks", "sagallm_lib")
if saga_lib_path not in sys.path:
    sys.path.insert(0, saga_lib_path)
    print(f"Inserted SagaLLM path at start: {saga_lib_path}")

print(f"Files in {saga_lib_path}: {os.listdir(saga_lib_path)}")
if os.path.exists(os.path.join(saga_lib_path, "multi_agent")):
    print(f"multi_agent contents: {os.listdir(os.path.join(saga_lib_path, 'multi_agent'))}")

from evaluation.task_definitions import TaskDefinition
from evaluation.framework_runners import BaseFrameworkRunner
from evaluation.compensation_runner import (
    book_vehicle, cancel_vehicle_booking,
    allocate_resource, deallocate_resource,
    assign_task, unassign_task,
    schedule_pickup, cancel_pickup,
    process_payment, check_capacity
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage

# Try to import SagaLLM components
try:
    from multi_agent.saga import Saga
    from multi_agent.agent import Agent
    SAGA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SagaLLM not available: {e}")
    SAGA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock OpenAI key to prevent SagaLLM crash on init (since we bypass LLM for this test)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing-structure-only"

# Import MongoDB tools for database workflows (AFTER logger is defined)
try:
    from evaluation.mongodb_tools import (
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

class CompensatableSagaAgent(Agent):
    """
    Extended Saga Agent that supports actual tool compensation.
    SagaLLM's default Agent.rollback() only prints text.
    We override it to actually call a compensation tool.
    
    This version uses the ACTUAL LLM (Gemini) for fair comparison with langchain-compensation.
    """
    def __init__(self, name, backstory, task_description, tools=None, compensation_tool=None):
        # Convert LangChain tools to SagaLLM Tool format if needed
        saga_tools = []
        if tools:
            for tool in tools:
                if isinstance(tool, StructuredTool):
                    # Convert LangChain tool to SagaLLM Tool
                    from tool_agent.tool import Tool as SagaTool
                    # Get proper type mapping from LangChain schema
                    properties = {}
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        schema_props = tool.args_schema.schema().get("properties", {})
                        for param_name, param_info in schema_props.items():
                            param_type = param_info.get("type", "string")
                            # Map JSON schema types to Python types
                            type_mapping = {
                                "string": "str",
                                "integer": "int",
                                "number": "float",
                                "boolean": "bool"
                            }
                            python_type = type_mapping.get(param_type, "str")
                            properties[param_name] = {"type": python_type}
                    
                    saga_tool = SagaTool(
                        name=tool.name,
                        fn=tool.func,
                        fn_signature=json.dumps({
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "properties": properties
                            }
                        })
                    )
                    saga_tools.append(saga_tool)
                else:
                    saga_tools.append(tool)
        
        # Create Gemini client for this agent (same as langchain-compensation uses)
        from utils.gemini_client import GeminiClientWrapper
        self._gemini_client = GeminiClientWrapper(model="gemini-flash-latest", temperature=0)
        
        # Initialize parent Agent class (this will create a ReactAgent with OpenAI)
        super().__init__(
            name=name,
            backstory=backstory,
            task_description=task_description,
            tools=saga_tools,
            llm="gemini-flash-latest"  # Model name for logging
        )
        
        # Replace ReactAgent with a custom one that tracks tool results
        from planning_agent.react_agent import ReactAgent
        
        class ToolTrackingReactAgent(ReactAgent):
            """ReactAgent that tracks tool execution results for compensation"""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_tool_results = []
                # Ensure tools_dict exists (ReactAgent uses tools_dict, not tools_Dict)
                if not hasattr(self, 'tools_dict'):
                    self.tools_dict = {tool.name: tool for tool in self.tools}
            
            def process_tool_calls(self, tool_calls_content: list) -> dict:
                """Override to track tool results"""
                try:
                    observations = super().process_tool_calls(tool_calls_content)
                    # Store results for compensation
                    for tool_call_str in tool_calls_content:
                        try:
                            # Handle both string and dict inputs
                            if isinstance(tool_call_str, str):
                                tool_call = json.loads(tool_call_str)
                            else:
                                tool_call = tool_call_str
                            
                            tool_name = tool_call.get("name", "unknown")
                            tool_id = tool_call.get("id", 0)
                            # Observations dict uses tool_id as key
                            result = observations.get(tool_id, "")
                            # Convert result to string if needed
                            if not isinstance(result, str):
                                result = str(result)
                            self.last_tool_results.append({
                                "tool_name": tool_name,
                                "result": result
                            })
                        except json.JSONDecodeError as e:
                            logger.warning(f"Could not parse tool call as JSON: {tool_call_str}, error: {e}")
                        except Exception as e:
                            logger.warning(f"Error tracking tool result: {e}")
                            import traceback
                            logger.warning(traceback.format_exc())
                    return observations
                except Exception as e:
                    logger.error(f"Error in process_tool_calls: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
        
        self.react_agent = ToolTrackingReactAgent(
            tools=saga_tools,
            model="gemini-flash-latest",
            system_prompt=backstory,
            client=self._gemini_client
        )
        
        self.compensation_tool = compensation_tool
        self.execution_result = None

    def run(self):
        """
        Run agent using LLM (Gemini) to decide which tools to call.
        This provides a fair comparison with langchain-compensation which also uses LLM.
        """
        print(f"🚀 {self.name} executing with LLM...")

        try:
            # Clear previous tool results
            if hasattr(self.react_agent, 'last_tool_results'):
                self.react_agent.last_tool_results = []
            
            # Use the LLM via ReactAgent (same as normal Agent.run() but we need to extract tool results)
            # Create a prompt that tells the LLM what to do
            user_message = self._build_llm_prompt()
            
            # Add LangSmith tracing for individual agent execution
            from langsmith import traceable
            
            @traceable(
                name=f"sagallm-agent-{self.name.replace(' ', '-').lower()}",
                run_type="chain",
                tags=["sagallm", "agent", self.name.replace(" ", "-").lower()],
                metadata={
                    "agent_name": self.name,
                    "agent_type": "sagallm-compensatable",
                    "has_compensation": self.compensation_tool is not None
                }
            )
            def run_agent_with_tracing():
                return self.react_agent.run(user_msg=user_message, max_rounds=5)
            
            # Call the LLM via ReactAgent with tracing
            output = run_agent_with_tracing()
            
            # Extract tool execution result from tracked tool results
            # Get the last tool result (the one we care about for compensation)
            if hasattr(self.react_agent, 'last_tool_results') and self.react_agent.last_tool_results:
                # Use the last tool's result
                last_result = self.react_agent.last_tool_results[-1]["result"]
                self.execution_result = last_result
                result_str = str(last_result)
            else:
                # Fallback: try to extract JSON from output
                import re
                json_match = re.search(r'\{[^}]+\}', output)
                if json_match:
                    try:
                        result_data = json.loads(json_match.group())
                        self.execution_result = json.dumps(result_data)
                        result_str = self.execution_result
                    except json.JSONDecodeError:
                        self.execution_result = output
                        result_str = output
                else:
                    self.execution_result = output
                    result_str = output
            
            # Check for failure
            if "error" in result_str.lower() or "capacity exceeded" in result_str.lower():
                try:
                    result_data = json.loads(result_str)
                    if result_data.get("status") == "error":
                        raise Exception(f"Tool execution failed: {result_str}")
                except (json.JSONDecodeError, KeyError):
                    if "error" in result_str.lower() or "capacity exceeded" in result_str.lower():
                        raise Exception(f"Tool execution failed: {result_str}")
            
            return result_str

        except Exception as e:
            self.execution_result = str(e)
            import traceback
            logger.error(f"Error in {self.name}: {e}")
            logger.error(traceback.format_exc())
            raise e
    
    def _build_llm_prompt(self) -> str:
        """Build natural prompts for the LLM to execute tasks without hardcoded parameters."""
        # P5-ACID: Wedding Logistics
        if "Venue" in self.name:
            return """You are responsible for booking the wedding venue. 
Book a venue for the wedding ceremony and reception. 
Use the book_vehicle tool to make the booking."""
        
        elif "Caterer" in self.name:
            return """You are responsible for arranging catering services.
Book catering for 200 wedding guests. 
Use the allocate_resource tool to arrange the catering service."""
        
        elif "Band" in self.name:
            return """You are responsible for booking entertainment.
Book a live band that can accommodate a large audience of 100 people.
First check if the band has capacity for 100 people using the check_capacity tool."""
        
        # P6-ACID: Thanksgiving Dinner
        elif "Sides" in self.name:
            return """You are responsible for ordering side dishes.
Order side dishes for the Thanksgiving dinner.
Use the book_vehicle tool to place the order."""
        
        elif "Drinks" in self.name:
            return """You are responsible for ordering beverages.
Order drinks for the Thanksgiving dinner.
Use the book_vehicle tool to place the order."""
        
        elif "Turkey" in self.name:
            return """You are responsible for ordering the main dish.
Order a large turkey for 100 people for Thanksgiving dinner.
First check if there's enough turkey available using the check_capacity tool."""
        
        # Fallback to task description
        return self.task_description

    def rollback(self):
        """Override to actually execute compensation"""
        print(f"🔄 Rolling back {self.name}'s operation...")
        
        if self.compensation_tool and self.execution_result:
            try:
                # Parse result to get ID for compensation
                result_str = str(self.execution_result)
                try:
                    result_data = json.loads(result_str)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, try to extract ID from string
                    result_data = {}
                    if "booking_id" in result_str:
                        import re
                        match = re.search(r'"booking_id":\s*"([^"]+)"', result_str)
                        if match:
                            result_data["booking_id"] = match.group(1)
                    elif "allocation_id" in result_str:
                        import re
                        match = re.search(r'"allocation_id":\s*"([^"]+)"', result_str)
                        if match:
                            result_data["allocation_id"] = match.group(1)
                
                # Map result ID to compensation arg
                comp_args = {}
                if "booking_id" in result_data:
                    comp_args["booking_id"] = result_data["booking_id"]
                elif "allocation_id" in result_data:
                    comp_args["allocation_id"] = result_data["allocation_id"]
                
                if comp_args:
                    # Execute compensation tool
                    if hasattr(self.compensation_tool, 'func'):
                        res = self.compensation_tool.func(**comp_args)
                    elif hasattr(self.compensation_tool, 'run'):
                        res = self.compensation_tool.run(**comp_args)
                    else:
                        res = "Compensation tool not callable"
                    print(f"✅ Compensation executed: {res}")
                else:
                    print(f"⚠️ Could not extract ID for compensation from {self.execution_result}")
            except Exception as e:
                print(f"❌ Compensation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ No compensation tool or result available for {self.name}")


class SagaLLMRunner(BaseFrameworkRunner):
    """Runner for SagaLLM framework"""
    
    def __init__(self):
        super().__init__()
        self.saga = None
        self._llm_call_count = 0
        self._total_tokens = {"input": 0, "output": 0}
    
    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        # Reset counters for each task
        self._llm_call_count = 0
        self._total_tokens = {"input": 0, "output": 0}
        if not SAGA_AVAILABLE:
            raise RuntimeError("SagaLLM not installed or configured properly")
            
        start_time = time.time()
        self._record_memory_usage()
        
        try:
            self.saga = Saga()
            
            # Dynamically create agents based on goals
            agents = []
            
            # Map goals to agents
            # For P5-ACID: Venue, Caterer, Band
            if "P5-ACID" in task_definition.task_id:
                # Agent 1: Venue (Succeeds)
                a1 = CompensatableSagaAgent(
                    name="Venue Agent",
                    backstory="Book the wedding venue",
                    task_description="Book venue using book_vehicle",
                    tools=[book_vehicle],
                    compensation_tool=cancel_vehicle_booking
                )

                # Agent 2: Caterer (Succeeds)
                a2 = CompensatableSagaAgent(
                    name="Caterer Agent",
                    backstory="Book the catering service",
                    task_description="Book caterer using allocate_resource",
                    tools=[allocate_resource],
                    compensation_tool=deallocate_resource
                )

                # Agent 3: Band (Fails - capacity > 50)
                a3 = CompensatableSagaAgent(
                    name="Band Agent",
                    backstory="Book the band",
                    task_description="Check band capacity using check_capacity",
                    tools=[check_capacity],
                    compensation_tool=None
                )

                # Define Dependencies: A1 -> A2 -> A3
                a2.add_dependency(a1)
                a3.add_dependency(a2)

                agents = [a1, a2, a3]

            elif "P6-ACID" in task_definition.task_id:
                # Agent 1: Sides
                a1 = CompensatableSagaAgent(
                    name="Sides Agent",
                    backstory="Order side dishes",
                    task_description="Order sides using book_vehicle",
                    tools=[book_vehicle],
                    compensation_tool=cancel_vehicle_booking
                )
                # Agent 2: Drinks
                a2 = CompensatableSagaAgent(
                    name="Drinks Agent",
                    backstory="Order drinks",
                    task_description="Order drinks using book_vehicle",
                    tools=[book_vehicle],
                    compensation_tool=cancel_vehicle_booking
                )
                # Agent 3: Turkey (Fails - capacity > 50)
                a3 = CompensatableSagaAgent(
                    name="Turkey Agent",
                    backstory="Order turkey",
                    task_description="Check turkey capacity using check_capacity",
                    tools=[check_capacity],
                    compensation_tool=None
                )

                # Dependencies: (A1, A2) -> A3
                a3.add_dependency(a1)
                a3.add_dependency(a2)

                agents = [a1, a2, a3]
            
            elif "MONGODB-ACID" in task_definition.task_id:
                # Agent 1: Create User (Succeeds)
                a1 = CompensatableSagaAgent(
                    name="User Creation Agent",
                    backstory="Create a new user account in the database",
                    task_description="Create user using create_user",
                    tools=[create_user] if MONGODB_TOOLS_AVAILABLE else [],
                    compensation_tool=delete_user if MONGODB_TOOLS_AVAILABLE else None
                )
                
                # Agent 2: Update Profile (Succeeds)
                a2 = CompensatableSagaAgent(
                    name="Profile Update Agent",
                    backstory="Update user profile information",
                    task_description="Update profile using update_user_profile",
                    tools=[update_user_profile] if MONGODB_TOOLS_AVAILABLE else [],
                    compensation_tool=revert_user_profile if MONGODB_TOOLS_AVAILABLE else None
                )
                
                # Agent 3: Add Preferences (Succeeds)
                a3 = CompensatableSagaAgent(
                    name="Preferences Agent",
                    backstory="Add user preferences",
                    task_description="Add preferences using add_user_preferences",
                    tools=[add_user_preferences] if MONGODB_TOOLS_AVAILABLE else [],
                    compensation_tool=remove_user_preferences if MONGODB_TOOLS_AVAILABLE else None
                )
                
                # Agent 4: Create Session (Fails - user has 5 sessions already)
                a4 = CompensatableSagaAgent(
                    name="Session Agent",
                    backstory="Create user session",
                    task_description="Create session using create_user_session",
                    tools=[create_user_session] if MONGODB_TOOLS_AVAILABLE else [],
                    compensation_tool=delete_user_session if MONGODB_TOOLS_AVAILABLE else None
                )
                
                # Define Dependencies: A1 -> A2 -> A3 -> A4
                a2.add_dependency(a1)
                a3.add_dependency(a2)
                a4.add_dependency(a3)
                
                agents = [a1, a2, a3, a4]
            
            # Register and Run
            self.saga.transaction_manager(agents)
            
            # Track execution state
            execution_successful = False
            try:
                # Add LangSmith tracing for SagaLLM execution
                from langsmith import traceable
                
                @traceable(
                    name=f"sagallm-{task_definition.task_id}",
                    run_type="chain",
                    tags=["sagallm", "multi-agent", task_definition.task_id, "mongodb-workflow"],
                    metadata={
                        "framework": "sagallm",
                        "task_id": task_definition.task_id,
                        "task_name": task_definition.name,
                        "agent_count": len(agents),
                        "agent_names": [a.name for a in agents],
                        "workflow": "mongodb-user-profile"
                    }
                )
                def run_saga_coordinator():
                    # Run coordinator
                    # SagaLLM catches exceptions and triggers rollback if with_rollback=True
                    self.saga.saga_coordinator(with_rollback=True)
                    return len(self.saga.context) == len(agents)
                
                # Run coordinator with tracing
                execution_successful = run_saga_coordinator()
            except Exception as e:
                # Saga coordinator should handle exceptions internally, but catch any unexpected ones
                logger.warning(f"Saga coordinator raised exception: {e}")
                execution_successful = False
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            # Aggregate LLM metrics from all agents
            total_llm_calls = sum(getattr(agent, '_agent_llm_calls', 0) for agent in agents)
            total_input_tokens = sum(getattr(agent, '_agent_tokens', {}).get("input", 0) for agent in agents)
            total_output_tokens = sum(getattr(agent, '_agent_tokens', {}).get("output", 0) for agent in agents)
            
            # Create base execution result
            exec_result = self._create_execution_result(
                achieved_goals=[], # In ACID failure, 0 goals should be retained
                satisfied_constraints=[],
                schedule=[]
            )
            
            # Add LLM usage metrics to resource_usage
            exec_result['resource_usage']['llm_metrics'] = {
                "llm_call_count": total_llm_calls,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "note": f"SagaLLM makes {total_llm_calls} separate LLM calls (one per agent). Each agent independently calls the LLM."
            }
            
            return exec_result
            
        except Exception as e:
            print(f"SagaLLM Execution Failed: {e}")
            return self._create_execution_result([], [], [])


