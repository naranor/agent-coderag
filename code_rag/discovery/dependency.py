import importlib
import inspect
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def extract_library_api(library_name: str) -> str:
    """
    Extracts the public API (classes, methods) of an installed library.
    """
    try:
        lib = importlib.import_module(library_name)
        output = [f"# Public API for '{library_name}':"]

        # Simple introspection
        for name, obj in inspect.getmembers(lib):
            if name.startswith("_"): continue

            if inspect.isclass(obj):
                output.append(f"- **Class: {name}**")
                # List methods
                for m_name, m_obj in inspect.getmembers(obj):
                    if m_name.startswith("_"): continue
                    if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                        try:
                            sig = inspect.signature(m_obj)
                            output.append(f"  - `{m_name}{sig}`")
                        except:
                            output.append(f"  - `{m_name}(...)`")
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                try:
                    sig = inspect.signature(obj)
                    output.append(f"- **Function: {name}{sig}**")
                except:
                    output.append(f"- **Function: {name}(...)**")

        return "\n".join(output[:100]) # Limit output length
    except Exception as e:
        return f"Failed to extract API for '{library_name}': {e}"
