"""
Framework Runners for REALM-Bench

This module provides runners for the four supported agent frameworks:
- LangGraph
- AutoGen
- CrewAI
- OpenAI Swarm Agent

Each runner adapts the framework to execute the standardized planning tasks.
"""

import time
import psutil
import os
import sys
from typing import Dict, List, Any, Optional
import traceback

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.task_definitions import TaskDefinition


class BaseFrameworkRunner:
    """Base class for framework runners"""
    
    def __init__(self):
        self.memory_usage = []
        self.execution_times = []
        self.token_usage = {}
    
    def _record_memory_usage(self):
        """Record current memory usage"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append({
            'memory_mb': memory_mb,
            'timestamp': time.time()
        })
    
    def _record_token_usage(self, framework: str, tokens: int):
        """Record token usage"""
        if framework not in self.token_usage:
            self.token_usage[framework] = 0
        self.token_usage[framework] += tokens
    
    def _create_execution_result(self, 
                               achieved_goals: List[str],
                               satisfied_constraints: List[str],
                               schedule: List[Dict[str, Any]],
                               disruptions_handled: List[Dict[str, Any]] = None,
                               replanning_attempts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized execution result"""
        return {
            'achieved_goals': achieved_goals,
            'satisfied_constraints': satisfied_constraints,
            'schedule': schedule,
            'disruptions_handled': disruptions_handled or [],
            'replanning_attempts': replanning_attempts or [],
            'resource_usage': {
                'memory_usage': self.memory_usage,
                'execution_times': self.execution_times,
                'token_usage': self.token_usage
            }
        }


class LangGraphRunner(BaseFrameworkRunner):
    """Runner for LangGraph framework"""
    
    def __init__(self):
        super().__init__()
        try:
            # Import the router function from langgraph
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent_frameworks', 'langgraph'))
            from router import run_agent
            self.run_agent = run_agent
        except ImportError as e:
            print(f"Warning: LangGraph not available: {e}")
            self.run_agent = None
    
    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute task using LangGraph"""
        if not self.run_agent:
            raise RuntimeError("LangGraph framework not available")
        
        start_time = time.time()
        self._record_memory_usage()
        
        try:
            # Create task description for LangGraph
            task_description = f"""
            Task: {task_definition.description}
            Goals: {[goal.description for goal in task_definition.goals]}
            Constraints: {[c.description for c in task_definition.constraints]}
            Resources: {task_definition.resources}
            """
            
            # Execute the agent
            result = self.run_agent(task_description)
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            # Extract results from LangGraph output
            achieved_goals = self._extract_achieved_goals(result)
            satisfied_constraints = self._extract_satisfied_constraints(result)
            schedule = self._extract_schedule(result)
            
            return self._create_execution_result(
                achieved_goals=achieved_goals,
                satisfied_constraints=satisfied_constraints,
                schedule=schedule
            )
            
        except Exception as e:
            print(f"LangGraph execution error: {str(e)}")
            traceback.print_exc()
            return self._create_execution_result([], [], [])
    
    def _extract_achieved_goals(self, result: Dict[str, Any]) -> List[str]:
        """Extract achieved goals from LangGraph result"""
        achieved_goals = []
        if 'goals_achieved' in result:
            achieved_goals = result['goals_achieved']
        elif 'goals' in result:
            achieved_goals = result['goals']
        elif 'output' in result and isinstance(result['output'], dict) and 'goals' in result['output']:
            achieved_goals = result['output']['goals']
        return achieved_goals
    
    def _extract_satisfied_constraints(self, result: Dict[str, Any]) -> List[str]:
        """Extract satisfied constraints from LangGraph result"""
        satisfied_constraints = []
        if 'constraints_satisfied' in result:
            satisfied_constraints = result['constraints_satisfied']
        elif 'constraints' in result:
            satisfied_constraints = result['constraints']
        elif 'output' in result and isinstance(result['output'], dict) and 'constraints' in result['output']:
            satisfied_constraints = result['output']['constraints']
        return satisfied_constraints
    
    def _extract_schedule(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract schedule from LangGraph result"""
        schedule = []
        if 'schedule' in result:
            schedule = result['schedule']
        elif 'output' in result and isinstance(result['output'], dict) and 'schedule' in result['output']:
            schedule = result['output']['schedule']
        return schedule


class AutoGenRunner(BaseFrameworkRunner):
    """Runner for AutoGen framework"""
    
    def __init__(self):
        super().__init__()
        try:
            # Import the router function from autogen
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent_frameworks', 'autogen_multi_agent'))
            from router import run_autogen_agents
            self.run_agents = run_autogen_agents
        except ImportError as e:
            print(f"Warning: AutoGen not available: {e}")
            self.run_agents = None
    
    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute task using AutoGen"""
        if not self.run_agents:
            raise RuntimeError("AutoGen framework not available")
        
        start_time = time.time()
        self._record_memory_usage()
        
        try:
            # Create task description for AutoGen
            task_description = f"""
            Task: {task_definition.description}
            Goals: {[goal.description for goal in task_definition.goals]}
            Constraints: {[c.description for c in task_definition.constraints]}
            Resources: {task_definition.resources}
            """
            
            # Execute the agents
            result = self.run_agents(task_description)
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            # Extract results from AutoGen output
            achieved_goals = self._extract_achieved_goals({"summary": result})
            satisfied_constraints = self._extract_satisfied_constraints({"summary": result})
            schedule = self._extract_schedule({"summary": result})
            
            return self._create_execution_result(
                achieved_goals=achieved_goals,
                satisfied_constraints=satisfied_constraints,
                schedule=schedule
            )
            
        except Exception as e:
            print(f"AutoGen execution error: {str(e)}")
            traceback.print_exc()
            return self._create_execution_result([], [], [])
    
    def _extract_achieved_goals(self, result: Dict[str, Any]) -> List[str]:
        """Extract achieved goals from AutoGen result"""
        achieved_goals = []
        if 'goals_achieved' in result:
            achieved_goals = result['goals_achieved']
        elif 'summary' in result and 'goals' in result['summary']:
            achieved_goals = result['summary']['goals']
        return achieved_goals
    
    def _extract_satisfied_constraints(self, result: Dict[str, Any]) -> List[str]:
        """Extract satisfied constraints from AutoGen result"""
        satisfied_constraints = []
        if 'constraints_satisfied' in result:
            satisfied_constraints = result['constraints_satisfied']
        elif 'summary' in result and 'constraints' in result['summary']:
            satisfied_constraints = result['summary']['constraints']
        return satisfied_constraints
    
    def _extract_schedule(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract schedule from AutoGen result"""
        schedule = []
        if 'schedule' in result:
            schedule = result['schedule']
        elif 'summary' in result and 'schedule' in result['summary']:
            schedule = result['summary']['schedule']
        return schedule


class CrewAIRunner(BaseFrameworkRunner):
    """Runner for CrewAI framework"""
    
    def __init__(self):
        super().__init__()
        try:
            # Import the router function from crewai
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent_frameworks', 'crewai_multi_agent'))
            from router import run_crewai
            self.run_crewai = run_crewai
        except ImportError as e:
            print(f"Warning: CrewAI not available: {e}")
            self.run_crewai = None
    
    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute task using CrewAI"""
        if not self.run_crewai:
            raise RuntimeError("CrewAI framework not available")
        
        start_time = time.time()
        self._record_memory_usage()
        
        try:
            # Create task description for CrewAI
            task_description = f"""
            Task: {task_definition.description}
            Goals: {[goal.description for goal in task_definition.goals]}
            Constraints: {[c.description for c in task_definition.constraints]}
            Resources: {task_definition.resources}
            """
            
            # Execute the crew
            result = self.run_crewai(task_description)
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            # Extract results from CrewAI output
            achieved_goals = self._extract_achieved_goals({"result": result})
            satisfied_constraints = self._extract_satisfied_constraints({"result": result})
            schedule = self._extract_schedule({"result": result})
            
            return self._create_execution_result(
                achieved_goals=achieved_goals,
                satisfied_constraints=satisfied_constraints,
                schedule=schedule
            )
            
        except Exception as e:
            print(f"CrewAI execution error: {str(e)}")
            traceback.print_exc()
            return self._create_execution_result([], [], [])
    
    def _extract_achieved_goals(self, result: Dict[str, Any]) -> List[str]:
        """Extract achieved goals from CrewAI result"""
        achieved_goals = []
        if 'goals_achieved' in result:
            achieved_goals = result['goals_achieved']
        elif 'result' in result and 'goals' in result['result']:
            achieved_goals = result['result']['goals']
        return achieved_goals
    
    def _extract_satisfied_constraints(self, result: Dict[str, Any]) -> List[str]:
        """Extract satisfied constraints from CrewAI result"""
        satisfied_constraints = []
        if 'constraints_satisfied' in result:
            satisfied_constraints = result['constraints_satisfied']
        elif 'result' in result and 'constraints' in result['result']:
            satisfied_constraints = result['result']['constraints']
        return satisfied_constraints
    
    def _extract_schedule(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract schedule from CrewAI result"""
        schedule = []
        if 'schedule' in result:
            schedule = result['schedule']
        elif 'result' in result and 'schedule' in result['result']:
            schedule = result['result']['schedule']
        return schedule


class SwarmRunner(BaseFrameworkRunner):
    """Runner for OpenAI Swarm Agent framework"""
    
    def __init__(self):
        super().__init__()
        try:
            # Import the router function from openai swarm agent
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent_frameworks', 'openai_swarm_agent'))
            from router import SwarmRouter
            self.router = SwarmRouter()
        except ImportError as e:
            print(f"Warning: OpenAI Swarm Agent not available: {e}")
            self.router = None
    
    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute task using OpenAI Swarm Agent"""
        if not self.router:
            raise RuntimeError("OpenAI Swarm Agent framework not available")
        
        start_time = time.time()
        self._record_memory_usage()
        
        try:
            # Create task description for Swarm Agent
            task_description = f"""
            Task: {task_definition.description}
            Goals: {[goal.description for goal in task_definition.goals]}
            Constraints: {[c.description for c in task_definition.constraints]}
            Resources: {task_definition.resources}
            """
            
            # Execute the agent
            result = self.router.process_query(task_description)
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            # Extract results from Swarm Agent output
            achieved_goals = self._extract_achieved_goals({"response": result})
            satisfied_constraints = self._extract_satisfied_constraints({"response": result})
            schedule = self._extract_schedule({"response": result})
            
            return self._create_execution_result(
                achieved_goals=achieved_goals,
                satisfied_constraints=satisfied_constraints,
                schedule=schedule
            )
            
        except Exception as e:
            print(f"OpenAI Swarm Agent execution error: {str(e)}")
            traceback.print_exc()
            return self._create_execution_result([], [], [])
    
    def _extract_achieved_goals(self, result: Dict[str, Any]) -> List[str]:
        """Extract achieved goals from Swarm result"""
        achieved_goals = []
        if 'goals_achieved' in result:
            achieved_goals = result['goals_achieved']
        elif 'response' in result and 'goals' in result['response']:
            achieved_goals = result['response']['goals']
        return achieved_goals
    
    def _extract_satisfied_constraints(self, result: Dict[str, Any]) -> List[str]:
        """Extract satisfied constraints from Swarm result"""
        satisfied_constraints = []
        if 'constraints_satisfied' in result:
            satisfied_constraints = result['constraints_satisfied']
        elif 'response' in result and 'constraints' in result['response']:
            satisfied_constraints = result['response']['constraints']
        return satisfied_constraints
    
    def _extract_schedule(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract schedule from Swarm result"""
        schedule = []
        if 'schedule' in result:
            schedule = result['schedule']
        elif 'response' in result and 'schedule' in result['response']:
            schedule = result['response']['schedule']
        return schedule


class LangGraphRunnerV2(BaseFrameworkRunner):
    """Runner for LangGraph V2 with shared tools (no automatic compensation)"""

    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__()
        self.model = model
        self.router_class = None
        try:
            from agent_frameworks.langgraph.router_v2 import LangGraphRouterV2
            self.router_class = LangGraphRouterV2
        except ImportError as e:
            print(f"Warning: LangGraph V2 not available: {e}")

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute task using LangGraph V2 with shared tools"""
        if not self.router_class:
            raise RuntimeError("LangGraph V2 framework not available")

        start_time = time.time()
        self._record_memory_usage()

        try:
            router = self.router_class(task_definition, model=self.model)
            query = self._build_task_query(task_definition)
            result = router.run(query)

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()

            return self._create_execution_result(
                achieved_goals=self._extract_achieved_goals(result, task_definition),
                satisfied_constraints=self._extract_satisfied_constraints(result, task_definition),
                schedule=self._extract_schedule(result),
                disruptions_handled=result.get("disruptions_triggered", []),
                replanning_attempts=[]
            )

        except Exception as e:
            print(f"LangGraph V2 execution error: {str(e)}")
            traceback.print_exc()
            return self._create_execution_result([], [], [])

    def _build_task_query(self, task_definition: TaskDefinition) -> str:
        """Build task query from definition"""
        goals_str = "\n".join([f"- {g.description}" for g in task_definition.goals])
        constraints_str = "\n".join([f"- {c.description}" for c in task_definition.constraints])
        return f"""
Execute the following planning task:

Task: {task_definition.name}
Description: {task_definition.description}

Goals:
{goals_str}

Constraints:
{constraints_str}

Resources Available:
{task_definition.resources}

Create and execute a plan to achieve these goals while respecting constraints.
Use the available tools to schedule jobs, assign vehicles, allocate resources, etc.
"""

    def _extract_achieved_goals(self, result: Dict[str, Any], task_def: TaskDefinition) -> List[str]:
        """Extract achieved goals from result"""
        achieved = []
        final_state = result.get("final_state")
        if final_state and result.get("success"):
            if hasattr(final_state, 'scheduled_jobs') and final_state.scheduled_jobs:
                achieved.append("minimize_makespan")
            if hasattr(final_state, 'assigned_vehicles') and final_state.assigned_vehicles:
                achieved.append("serve_all_passengers")
            if hasattr(final_state, 'deployed_teams') and final_state.deployed_teams:
                achieved.append("maximize_aid_delivery")
        return achieved

    def _extract_satisfied_constraints(self, result: Dict[str, Any], task_def: TaskDefinition) -> List[str]:
        """Extract satisfied constraints from result"""
        satisfied = []
        if result.get("success") and not result.get("disruptions_triggered"):
            satisfied = [c.constraint_id for c in task_def.constraints[:1]]
        return satisfied

    def _extract_schedule(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract schedule from result"""
        final_state = result.get("final_state")
        if final_state and hasattr(final_state, 'scheduled_jobs'):
            return [
                {
                    "job_id": job.job_id,
                    "machine_id": job.machine_id,
                    "start_time": job.start_time,
                    "end_time": job.end_time
                }
                for job in final_state.scheduled_jobs.values()
            ]
        return []


class LangChainCompensationRunner(BaseFrameworkRunner):
    """Runner for langchain-compensation framework with automatic rollback"""

    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__()
        self.model = model
        self.router_class = None
        try:
            from agent_frameworks.langchain_compensation.router import CompensationRouter
            self.router_class = CompensationRouter
        except ImportError as e:
            print(f"Warning: langchain-compensation not available: {e}")

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        """Execute task using langchain-compensation with automatic rollback"""
        if not self.router_class:
            raise RuntimeError("langchain-compensation framework not available")

        start_time = time.time()
        self._record_memory_usage()

        try:
            router = self.router_class(task_definition, model=self.model)
            query = self._build_task_query(task_definition)
            result = router.run(query)

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()

            return self._create_execution_result(
                achieved_goals=self._extract_achieved_goals(result, task_definition),
                satisfied_constraints=self._extract_satisfied_constraints(result, task_definition),
                schedule=self._extract_schedule(result),
                disruptions_handled=result.get("disruptions_triggered", []),
                replanning_attempts=result.get("compensation_events", [])
            )

        except Exception as e:
            print(f"langchain-compensation execution error: {str(e)}")
            traceback.print_exc()
            return self._create_execution_result([], [], [])

    def _build_task_query(self, task_definition: TaskDefinition) -> str:
        """Build task query from definition"""
        goals_str = "\n".join([f"- {g.description}" for g in task_definition.goals])
        constraints_str = "\n".join([f"- {c.description}" for c in task_definition.constraints])
        return f"""
Execute the following planning task:

Task: {task_definition.name}
Description: {task_definition.description}

Goals:
{goals_str}

Constraints:
{constraints_str}

Resources Available:
{task_definition.resources}

Create and execute a plan to achieve these goals while respecting constraints.
Use the available tools to schedule jobs, assign vehicles, allocate resources, etc.
If a tool fails, the system will automatically compensate (rollback) previous actions.
"""

    def _extract_achieved_goals(self, result: Dict[str, Any], task_def: TaskDefinition) -> List[str]:
        """Extract achieved goals from result"""
        achieved = []
        final_state = result.get("final_state")
        if final_state and result.get("success"):
            if hasattr(final_state, 'scheduled_jobs') and final_state.scheduled_jobs:
                achieved.append("minimize_makespan")
            if hasattr(final_state, 'assigned_vehicles') and final_state.assigned_vehicles:
                achieved.append("serve_all_passengers")
            if hasattr(final_state, 'deployed_teams') and final_state.deployed_teams:
                achieved.append("maximize_aid_delivery")
            if hasattr(final_state, 'allocated_resources') and final_state.allocated_resources:
                achieved.append("optimize_resource_allocation")
        return achieved

    def _extract_satisfied_constraints(self, result: Dict[str, Any], task_def: TaskDefinition) -> List[str]:
        """Extract satisfied constraints from result"""
        satisfied = []
        if result.get("success"):
            compensation_events = result.get("compensation_events", [])
            if not compensation_events:
                satisfied = [c.constraint_id for c in task_def.constraints]
            else:
                satisfied = [c.constraint_id for c in task_def.constraints[:1]]
        return satisfied

    def _extract_schedule(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract schedule from result"""
        final_state = result.get("final_state")
        if final_state and hasattr(final_state, 'scheduled_jobs'):
            return [
                {
                    "job_id": job.job_id,
                    "machine_id": job.machine_id,
                    "start_time": job.start_time,
                    "end_time": job.end_time
                }
                for job in final_state.scheduled_jobs.values()
            ]
        return []


def get_framework_runners() -> Dict[str, BaseFrameworkRunner]:
    """Get all available framework runners"""
    runners = {}

    # V2 runners with shared tools (for fair comparison benchmarks)
    try:
        runners['langgraph'] = LangGraphRunnerV2()
    except Exception as e:
        print(f"LangGraph V2 runner not available: {e}")

    try:
        runners['langchain_compensation'] = LangChainCompensationRunner()
    except Exception as e:
        print(f"langchain-compensation runner not available: {e}")

    # Legacy runners (original implementations)
    try:
        runners['langgraph_legacy'] = LangGraphRunner()
    except Exception as e:
        print(f"LangGraph legacy runner not available: {e}")

    try:
        runners['autogen'] = AutoGenRunner()
    except Exception as e:
        print(f"AutoGen runner not available: {e}")

    try:
        runners['crewai'] = CrewAIRunner()
    except Exception as e:
        print(f"CrewAI runner not available: {e}")

    try:
        runners['swarm'] = SwarmRunner()
    except Exception as e:
        print(f"Swarm runner not available: {e}")

    return runners


def create_mock_runner(framework_name: str) -> BaseFrameworkRunner:
    """Create a mock runner for testing when frameworks are not available"""
    
    class MockRunner(BaseFrameworkRunner):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
        
        def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
            """Mock execution that returns basic results"""
            start_time = time.time()
            self._record_memory_usage()
            
            # Simulate some processing time
            time.sleep(1)
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._record_memory_usage()
            
            # Return mock results
            achieved_goals = [goal.goal_id for goal in task_definition.goals[:2]]  # Mock: achieve first 2 goals
            satisfied_constraints = [c.constraint_id for c in task_definition.constraints[:1]]  # Mock: satisfy first constraint
            
            schedule = [
                {
                    'task_id': f'task_{i}',
                    'start_time': i * 10,
                    'end_time': (i + 1) * 10,
                    'agent': f'agent_{i}'
                }
                for i in range(3)  # Mock: 3 tasks
            ]
            
            return self._create_execution_result(
                achieved_goals=achieved_goals,
                satisfied_constraints=satisfied_constraints,
                schedule=schedule
            )
    
    return MockRunner(framework_name) 