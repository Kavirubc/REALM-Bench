
import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Add SagaLLM lib path explicitly before import
saga_lib_path = os.path.join(project_root, "agent_frameworks", "sagallm_lib")
sys.path.insert(0, saga_lib_path)

from evaluation.framework_runners import get_framework_runners

print("Getting runners...")
runners = get_framework_runners()
print(f"Runners: {list(runners.keys())}")

