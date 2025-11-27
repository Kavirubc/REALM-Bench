"""
Multi-Agent Compensation Framework Runner using langchain-compensation library.

This runner creates multiple agents (similar to SagaLLM) where each agent handles
a specific task, but all agents share compensation middleware for coordinated rollback.
"""

import os
import sys
import time
import logging
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "realm-bench-compensation-multiagent")

# Ensure LangSmith API key is set (if available)
if "LANGSMITH_API_KEY" not in os.environ:
    # Try to get from .env file
    from dotenv import load_dotenv
    load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent
from langchain_compensation.middleware import CompensationLog
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

# Update compensation mapping if MongoDB tools available
if MONGODB_TOOLS_AVAILABLE:
    COMPENSATION_MAPPING.update(MONGODB_COMPENSATION_MAPPING)


class MultiAgentCompensationRunner(BaseFrameworkRunner):
    """
    Multi-agent runner using langchain-compensation.
    
    Creates separate agents for each task (like SagaLLM), but each agent
    uses langchain-compensation's create_comp_agent. All agents share a
    compensation log for coordinated rollback.
    """

    def __init__(self):
        super().__init__()
        
        # Initialize LLM (same as SagaLLM uses)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0
        )
        
        logger.info("MultiAgentCompensationRunner initialized")

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        return self.call(task_definition)

    def call(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute a task using multiple agents with shared compensation."""
        start_time = time.time()
        self._record_memory_usage()
        
        task_id = task_definition.task_id
        
        try:
            # Create shared compensation log for all agents
            shared_comp_log = CompensationLog()
            
            # Create agents based on task (all share the same comp_log_ref)
            agents = self._create_agents(task_id, shared_comp_log)
            
            # Execute agents in order (with dependencies like SagaLLM)
            execution_results = []
            executed_agents = []
            
            for agent_info in agents:
                agent_name = agent_info["name"]
                agent = agent_info["agent"]
                task_prompt = agent_info["prompt"]
                dependencies = agent_info.get("dependencies", [])
                
                # Check if dependencies are satisfied
                if dependencies:
                    dep_results = [execution_results[i] for i in dependencies if i < len(execution_results)]
                    if len(dep_results) < len(dependencies):
                        logger.warning(f"{agent_name} dependencies not satisfied, skipping")
                        continue
                
                logger.info(f"Executing {agent_name}...")
                
                try:
                    # Execute agent (compensation log is already shared via comp_log_ref in create_comp_agent)
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=task_prompt)]},
                        config={
                            "run_name": f"compensation-multiagent-{task_id}-{agent_name.replace(' ', '-').lower()}",
                            "tags": ["compensation-multiagent", "multi-agent", task_id, agent_name.replace(" ", "-").lower(), "mongodb-workflow"],
                            "metadata": {
                                "framework": "compensation_multiagent",
                                "task_id": task_id,
                                "agent_name": agent_name,
                                "agent_index": len(executed_agents),
                                "workflow": "mongodb-user-profile"
                            }
                        }
                    )
                    
                    # Extract tool execution result
                    agent_result = self._extract_agent_result(result)
                    
                    # Check messages for error status (middleware sets status="error" on ToolMessage)
                    has_error = False
                    for msg in result.get("messages", []):
                        if hasattr(msg, 'status') and msg.status == "error":
                            has_error = True
                            logger.warning(f"Found error status in message: {msg}")
                            break
                        if hasattr(msg, 'content'):
                            if isinstance(msg.content, dict) and msg.content.get("status") == "error":
                                has_error = True
                                break
                            if isinstance(msg.content, str):
                                # Check if it's a JSON error
                                try:
                                    import re
                                    json_match = re.search(r'\{[^}]+\}', msg.content)
                                    if json_match:
                                        json_data = json.loads(json_match.group())
                                        if json_data.get("status") == "error":
                                            has_error = True
                                            break
                                except:
                                    pass
                    
                    # Check compensation log to see if rollback happened
                    rollback_plan = shared_comp_log.get_rollback_plan()
                    if rollback_plan:
                        logger.info(f"⚠️ Compensation log shows {len(rollback_plan)} actions need rollback")
                        for record in rollback_plan:
                            logger.info(f"  - {record['tool_name']} (ID: {record['id']})")
                    
                    if has_error or self._is_failure(agent_result):
                        logger.error(f"{agent_name} failed: {agent_result}")
                        # Check if rollback already happened (middleware handles it automatically)
                        compensated_count = sum(1 for r in shared_comp_log.to_dict().values() if r.get("compensated", False))
                        if compensated_count > 0:
                            logger.info(f"✅ Compensation middleware already rolled back {compensated_count} actions")
                        # Stop execution here
                        raise Exception(f"{agent_name} execution failed: {agent_result}")
                    
                    execution_results.append(agent_result)
                    executed_agents.append(agent_info)
                    logger.info(f"{agent_name} completed successfully")
                    
                except Exception as e:
                    logger.error(f"{agent_name} error: {e}")
                    # Compensation middleware should have already handled rollback
                    # when it detected the error. We just propagate the exception.
                    raise
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            logger.info(f"All agents completed in {execution_time:.2f}s")
            
            return self._create_execution_result([], [], [])
            
        except Exception as e:
            logger.error(f"Multi-agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_execution_result([], [], [])

    def _create_agents(self, task_id: str, shared_comp_log: CompensationLog) -> List[Dict[str, Any]]:
        """Create agents for the given task, all sharing the same compensation log."""
        agents = []
        
        if task_id == "P5-ACID":
            # Agent 1: Venue
            venue_agent = create_comp_agent(
                model=self.llm,
                tools=[book_vehicle, cancel_vehicle_booking],
                compensation_mapping=COMPENSATION_MAPPING,
                system_prompt="""You are responsible for booking the wedding venue.
Use the book_vehicle tool to book a venue for the wedding ceremony and reception.""",
                comp_log_ref=shared_comp_log  # Share compensation log across all agents
            )
            agents.append({
                "name": "Venue Agent",
                "agent": venue_agent,
                "prompt": "Book a venue for the wedding ceremony and reception.",
                "tools": [book_vehicle],
                "dependencies": []
            })
            
            # Agent 2: Caterer
            caterer_agent = create_comp_agent(
                model=self.llm,
                tools=[allocate_resource, deallocate_resource],
                compensation_mapping=COMPENSATION_MAPPING,
                system_prompt="""You are responsible for arranging catering services.
Use the allocate_resource tool to arrange catering for 200 wedding guests.""",
                comp_log_ref=shared_comp_log  # Share compensation log
            )
            agents.append({
                "name": "Caterer Agent",
                "agent": caterer_agent,
                "prompt": "Arrange catering services for 200 wedding guests.",
                "tools": [allocate_resource],
                "dependencies": [0]  # Depends on Venue Agent
            })
            
            # Agent 3: Band (will fail)
            band_agent = create_comp_agent(
                model=self.llm,
                tools=[check_capacity],
                compensation_mapping={},  # No compensation for check_capacity
                system_prompt="""You are responsible for booking entertainment.
Book a live band that can accommodate a large audience of 100 people.
First check if the band has capacity for 100 people using the check_capacity tool.""",
                comp_log_ref=shared_comp_log  # Share compensation log (for rollback when this fails)
            )
            agents.append({
                "name": "Band Agent",
                "agent": band_agent,
                "prompt": "Book a live band for entertainment that can accommodate 100 people. First check capacity.",
                "tools": [check_capacity],
                "dependencies": [1]  # Depends on Caterer Agent
            })
            
        elif task_id == "P6-ACID":
            # Agent 1: Sides
            sides_agent = create_comp_agent(
                model=self.llm,
                tools=[book_vehicle, cancel_vehicle_booking],
                compensation_mapping=COMPENSATION_MAPPING,
                system_prompt="""You are responsible for ordering side dishes.
Order side dishes for the Thanksgiving dinner using the book_vehicle tool.""",
                comp_log_ref=shared_comp_log  # Share compensation log
            )
            agents.append({
                "name": "Sides Agent",
                "agent": sides_agent,
                "prompt": "Order side dishes for the Thanksgiving dinner.",
                "tools": [book_vehicle],
                "dependencies": []
            })
            
            # Agent 2: Drinks
            drinks_agent = create_comp_agent(
                model=self.llm,
                tools=[book_vehicle, cancel_vehicle_booking],
                compensation_mapping=COMPENSATION_MAPPING,
                system_prompt="""You are responsible for ordering beverages.
Order drinks for the Thanksgiving dinner using the book_vehicle tool.""",
                comp_log_ref=shared_comp_log  # Share compensation log
            )
            agents.append({
                "name": "Drinks Agent",
                "agent": drinks_agent,
                "prompt": "Order drinks and beverages for the Thanksgiving dinner.",
                "tools": [book_vehicle],
                "dependencies": []
            })
            
            # Agent 3: Turkey (will fail)
            turkey_agent = create_comp_agent(
                model=self.llm,
                tools=[check_capacity],
                compensation_mapping={},  # No compensation for check_capacity
                system_prompt="""You are responsible for ordering the main dish.
Order a large turkey for 100 people for Thanksgiving dinner.
First check if there's enough turkey available using the check_capacity tool.""",
                comp_log_ref=shared_comp_log  # Share compensation log (for rollback when this fails)
            )
            agents.append({
                "name": "Turkey Agent",
                "agent": turkey_agent,
                "prompt": "Order a large turkey for 100 people. First check if there's enough turkey available.",
                "tools": [check_capacity],
                "dependencies": [0, 1]  # Depends on both Sides and Drinks
            })
        
        elif task_id == "MONGODB-ACID":
            if not MONGODB_TOOLS_AVAILABLE:
                logger.error("MongoDB tools not available for MONGODB-ACID task")
                return []
            
            # Agent 1: Create User
            user_agent = create_comp_agent(
                model=self.llm,
                tools=[create_user, delete_user],
                compensation_mapping=MONGODB_COMPENSATION_MAPPING,
                system_prompt="""You are responsible for creating new user accounts.
Create a new user account in the database with the provided user information.""",
                comp_log_ref=shared_comp_log
            )
            # Get user_id from task resources
            user_id = "test_user_123"  # Default from task definition
            
            agents.append({
                "name": "User Creation Agent",
                "agent": user_agent,
                "prompt": f"Create a new user account in the database with user_id: {user_id}.",
                "tools": [create_user],
                "dependencies": []
            })
            
            # Agent 2: Update Profile
            profile_agent = create_comp_agent(
                model=self.llm,
                tools=[update_user_profile, revert_user_profile],
                compensation_mapping=MONGODB_COMPENSATION_MAPPING,
                system_prompt="""You are responsible for updating user profiles.
Update the user's profile with additional information like bio and location.""",
                comp_log_ref=shared_comp_log
            )
            agents.append({
                "name": "Profile Update Agent",
                "agent": profile_agent,
                "prompt": f"Update the user's profile with additional information for user_id: {user_id}.",
                "tools": [update_user_profile],
                "dependencies": [0]  # Depends on User Creation
            })
            
            # Agent 3: Add Preferences
            prefs_agent = create_comp_agent(
                model=self.llm,
                tools=[add_user_preferences, remove_user_preferences],
                compensation_mapping=MONGODB_COMPENSATION_MAPPING,
                system_prompt="""You are responsible for managing user preferences.
Add user preferences for the application such as theme and notification settings.""",
                comp_log_ref=shared_comp_log
            )
            agents.append({
                "name": "Preferences Agent",
                "agent": prefs_agent,
                "prompt": f"Add user preferences for the application for user_id: {user_id}.",
                "tools": [add_user_preferences],
                "dependencies": [1]  # Depends on Profile Update
            })
            
            # Agent 4: Create Session (will fail)
            session_agent = create_comp_agent(
                model=self.llm,
                tools=[create_user_session, delete_user_session],
                compensation_mapping=MONGODB_COMPENSATION_MAPPING,
                system_prompt="""You are responsible for creating user sessions.
Create an active session for the user to track their current session.""",
                comp_log_ref=shared_comp_log
            )
            agents.append({
                "name": "Session Agent",
                "agent": session_agent,
                "prompt": f"Create an active session for the user with user_id: {user_id}. Note: This user may already have active sessions.",
                "tools": [create_user_session],
                "dependencies": [2]  # Depends on Preferences
            })
        
        return agents

    def _extract_agent_result(self, result: Dict[str, Any]) -> str:
        """Extract the result from agent execution."""
        messages = result.get("messages", [])
        
        # Look for tool messages with results (especially error status)
        for msg in reversed(messages):
            # Check for ToolMessage with error status
            if hasattr(msg, 'status') and msg.status == "error":
                return json.dumps({"status": "error", "message": str(msg.content)})
            
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                
                # Check if content indicates error
                if isinstance(msg.content, dict) and msg.content.get("status") == "error":
                    return json.dumps(msg.content)
                
                # Try to extract JSON result
                try:
                    # Look for JSON in the content
                    import re
                    json_match = re.search(r'\{[^}]+\}', content)
                    if json_match:
                        json_str = json_match.group()
                        # Try to parse and check for error
                        try:
                            json_data = json.loads(json_str)
                            if json_data.get("status") == "error":
                                return json_str
                        except:
                            pass
                        return json_str
                except:
                    pass
                return content
        
        return "No result found"

    def _is_failure(self, result: str) -> bool:
        """Check if result indicates failure."""
        result_lower = result.lower()
        if "error" in result_lower or "capacity exceeded" in result_lower:
            try:
                result_data = json.loads(result)
                if result_data.get("status") == "error":
                    return True
            except:
                pass
            return True
        return False

    def _rollback_all(self, comp_log: CompensationLog, executed_agents: List[Dict[str, Any]]):
        """Trigger rollback on shared compensation log."""
        logger.info("🔄 Rolling back all executed agents...")
        
        # Get rollback plan from shared log
        rollback_plan = comp_log.get_rollback_plan()
        
        logger.info(f"Found {len(rollback_plan)} actions to compensate")
        for record in rollback_plan:
            logger.info(f"Compensating: {record['tool_name']}")
            comp_log.mark_compensated(record['id'])

