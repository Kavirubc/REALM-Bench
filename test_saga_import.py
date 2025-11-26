
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(project_root, "agent_frameworks", "sagallm_lib")
sys.path.append(lib_path)

print(f"Added to path: {lib_path}")
try:
    import multi_agent
    print(f"Imported multi_agent from: {multi_agent.__file__}")
    from multi_agent.saga import Saga
    print("Imported Saga successfully")
except Exception as e:
    print(f"Error: {e}")

