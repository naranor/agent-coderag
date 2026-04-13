import ast
import logging
import importlib.util
import os
from pathlib import Path

logger = logging.getLogger(__name__)

async def extract_library_api(library_name: str) -> str:
    """
    Finds the library path and parses its public members to create a summary.
    This function is used by the 'api' CLI command.
    """
    spec = importlib.util.find_spec(library_name)
    if not spec or not spec.origin:
        return f"Library {library_name} not found."

    lib_path = Path(spec.origin).parent
    output = [f"# Public API for {library_name} ({lib_path})"]
    
    # Simple top-level scan
    for py_file in lib_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not node.name.startswith("_"):
                        output.append(f"- {node.name}")
        except (OSError, SyntaxError) as e:
            logger.debug("Failed to parse %s: %s", py_file, e)

    return "\n".join(output)
