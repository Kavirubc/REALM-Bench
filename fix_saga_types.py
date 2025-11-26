
import os
import re

def fix_type_hints(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Add imports if missing
    if "from typing" not in content:
        content = "from typing import List, Optional, Union, Dict, Any\n" + content
    else:
        # Update existing typing import or add if it doesn't cover everything
        if "List" not in content: content = content.replace("from typing import", "from typing import List,")
        if "Optional" not in content: content = content.replace("from typing import", "from typing import Optional,")
        if "Union" not in content: content = content.replace("from typing import", "from typing import Union,")

    # Replace X | None with Optional[X]
    # This regex handles simple cases like "str | None" or "list | None"
    # It might struggle with nested ones, but let's try coverage
    
    # Replace "list[X]" with "List[X]"
    content = re.sub(r'list\[', 'List[', content)
    content = re.sub(r'dict\[', 'Dict[', content)
    
    # Replace "Type | None" with "Optional[Type]"
    # We need to be careful not to break bitwise OR
    # Look for ": Type | None" or "-> Type | None" patterns in function signatures
    
    def replace_optional(match):
        type_name = match.group(1)
        return f"Optional[{type_name}]"

    # Pattern:  : Something | None
    content = re.sub(r':\s*([a-zA-Z0-9_\[\].]+)\s*\|\s*None', r': Optional[\1]', content)
    # Pattern:  : None | Something
    content = re.sub(r':\s*None\s*\|\s*([a-zA-Z0-9_\[\].]+)', r': Optional[\1]', content)
    
    # Pattern: -> Something | None
    content = re.sub(r'->\s*([a-zA-Z0-9_\[\].]+)\s*\|\s*None', r'-> Optional[\1]', content)

    if content != original_content:
        print(f"Patching {file_path}")
        with open(file_path, 'w') as f:
            f.write(content)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                fix_type_hints(os.path.join(root, file))

if __name__ == "__main__":
    target_dir = "agent_frameworks/sagallm_lib"
    print(f"Scanning {target_dir}...")
    process_directory(target_dir)
    print("Done.")

