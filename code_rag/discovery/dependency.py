import importlib
import inspect
import logging
from .java_discovery import extract_java_api

logger = logging.getLogger(__name__)

def _get_method_signature(obj) -> str:
    """Helper to safely get a signature string."""
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "(...)"

async def extract_library_api(library_name: str) -> str:
    """
    Extracts the public API (classes, methods) of an installed library.
    Tries Python first, then falls back to Java discovery in local caches.
    """
    # 1. Try Python
    try:
        # Check if it's a known python module first without importing if possible
        # or just try to import.
        lib = importlib.import_module(library_name)
        output = [f"# Public API for Python Library '{library_name}':"]

        for name, obj in inspect.getmembers(lib):
            if name.startswith("_"):
                continue

            if inspect.isclass(obj):
                output.append(f"- **Class: {name}**")
                for m_name, m_obj in inspect.getmembers(obj):
                    if m_name.startswith("_"):
                        continue
                    if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                        sig = _get_method_signature(m_obj)
                        output.append(f"  - `{m_name}{sig}`")
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                sig = _get_method_signature(obj)
                output.append(f"- **Function: {name}{sig}**")

        return "\n".join(output[:100])
    except ImportError:
        # 2. If Python import fails, try Java discovery
        logger.info("Python import failed for '%s', trying Java discovery...", library_name)
        return await extract_java_api(library_name)
    except Exception as e:
        return f"Failed to extract API for '{library_name}': {e}"
