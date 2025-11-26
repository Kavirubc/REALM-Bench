"""
Compensation Framework Runner using langchain-compensation (direct usage)

This runner demonstrates how to use the langchain-compensation library as intended,
without embedding or adapting its logic. It is designed for comparison with the custom
integration in compensation_runner.py.
"""

import os
import sys
import logging
from typing import Any, Dict, List
from dotenv import load_dotenv

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import CompensationMiddleware, CompensationConfig
from .task_definitions import TaskDefinition
from .framework_runners import BaseFrameworkRunner

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example tools (should be replaced with real ones or imported from existing modules)
@tool
def book_vehicle(vehicle_id: str, passenger_id: str, route: str) -> str:
    """Books a vehicle for a passenger on a specific route. Returns booking_id."""
    return f"booking_{vehicle_id}_{passenger_id}"

@tool
def cancel_vehicle_booking(booking_id: str) -> str:
    """Cancels a vehicle booking. Returns cancellation status."""
    return f"cancelled_{booking_id}"

# Compensation mapping for tools
tool_compensation_map = {
    "book_vehicle": "cancel_vehicle_booking",
}

class CompensationLibRunner(BaseFrameworkRunner):
    """Framework runner that uses langchain-compensation middleware directly."""
    def __init__(self, tools: List[Any] = None):
        super().__init__()
        self.tools = tools or [book_vehicle, cancel_vehicle_booking]
        self.comp_config = CompensationConfig(
            compensation_map=tool_compensation_map,
            rollback_strategy="lifo"  # or "dag" if supported
        )
        self.middleware = CompensationMiddleware(self.comp_config)
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            middleware=[self.middleware],
            system_prompt="You are a helpful multi-agent planning assistant. Use tools as needed."
        )

    def call(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        user_message = task_definition.description if hasattr(task_definition, 'description') else "Execute the planning task."
        logger.info("Invoking agent with Gemini and compensation middleware...")
        result = self.agent.invoke({
            "messages": [
                {"role": "user", "content": user_message}
            ]
        })
        logger.info(f"Agent result: {result}")
        return {"status": "done", "result": result}

    def __call__(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        return self.call(task_definition)
