import importlib.util
import importlib.metadata
import inspect
import logging
from pathlib import Path
from typing import List, Optional

from .base import IDiscoveryProvider
from ...parsers.tree_sitter import TreeSitterParser
from ...core.exceptions import DiscoveryError

logger = logging.getLogger(__name__)


class PythonDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Python libraries using 3-stage probe."""

    def __init__(self, parser: Optional[TreeSitterParser] = None):
        self.parser = parser or TreeSitterParser()

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts API using:
        1. Static analysis of .pyi files (if found)
        2. Static analysis of .py files (if found)
        3. Runtime introspection (fallback)
        """
        try:
            # 1. Find the library path without importing
            lib_root = self._find_library_root(library_name)

            if lib_root:
                # Stage 1: Try .pyi files
                stubs = list(lib_root.rglob("*.pyi"))
                if stubs:
                    logger.info(
                        "Found %d type stubs for '%s'", len(stubs), library_name
                    )
                    return await self._extract_static(library_name, stubs)

                # Stage 2: Try .py files (top-level)
                # We look for __init__.py and files in the root
                sources = list(lib_root.glob("*.py"))
                if sources:
                    logger.info(
                        "Performing static analysis on '%s' source files", library_name
                    )
                    return await self._extract_static(library_name, sources)

            # Stage 3: Runtime Fallback
            logger.info("Falling back to runtime introspection for '%s'", library_name)
            return self._extract_runtime(library_name)
        except Exception as e:
            if isinstance(e, DiscoveryError):
                raise
            raise DiscoveryError(
                f"Failed to extract Python API for '{library_name}': {e}"
            ) from e

    def _find_library_root(self, lib_name: str) -> Optional[Path]:
        """Locates the package root without importing it."""
        try:
            # 1. Try find_spec (Standard way)
            spec = importlib.util.find_spec(lib_name)
            if spec and spec.origin:
                origin = Path(spec.origin)
                # If it's a package, origin is __init__.py, we want the directory
                return origin.parent if spec.submodule_search_locations else origin

            # 2. Try metadata (for pip-installed packages)
            dist_files = importlib.metadata.files(lib_name)
            if dist_files:
                for f in dist_files:
                    path = Path(f.locate())
                    if path.name == "__init__.py" and path.parent.name == lib_name:
                        return path.parent
                    if path.suffix == ".py" and path.stem == lib_name:
                        return path
        except Exception as e:
            logger.debug("Failed to find library root for %s: %s", lib_name, e)
        return None

    async def _extract_static(self, library_name: str, files: List[Path]) -> str:
        """Uses TreeSitterParser to extract API from files."""
        output = [
            f"# Public API for Python Library '{library_name}' (Static Analysis):"
        ]

        # Sort files to prioritize __init__.py and then by name
        sorted_files = sorted(files, key=lambda p: (p.name != "__init__.py", p.name))

        # Process up to 5 main files to keep it concise
        for file_path in sorted_files[:5]:
            try:
                units = await self.parser.distill_file(str(file_path))
                if units:
                    output.append(f"\n## From {file_path.name}:")
                    for unit in units:
                        # Only show top-level or interesting units
                        if "." not in unit.id.split(":")[-1] or unit.kind == "class":
                            output.append(
                                f"- **{unit.kind.value.capitalize()}: {unit.name}**"
                            )
                            output.append(f"  `{unit.signature}`")
            except Exception as e:
                logger.warning("Failed to parse %s: %s", file_path, e)

        return "\n".join(output[:100])

    def _extract_runtime(self, library_name: str) -> str:
        """Legacy runtime introspection logic."""
        try:
            lib = importlib.import_module(library_name)
            output = [f"# Public API for Python Library '{library_name}' (Runtime):"]

            for name, obj in inspect.getmembers(lib):
                if name.startswith("_"):
                    continue

                if inspect.isclass(obj):
                    output.append(f"- **Class: {name}**")
                    for m_name, m_obj in inspect.getmembers(obj):
                        if m_name.startswith("_"):
                            continue
                        if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                            sig = self._get_method_signature(m_obj)
                            output.append(f"  - `{m_name}{sig}`")
                elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    sig = self._get_method_signature(obj)
                    output.append(f"- **Function: {name}{sig}**")

            return "\n".join(output[:100])
        except Exception as e:
            return f"Failed to extract API for '{library_name}': {e}"

    def _get_method_signature(self, obj) -> str:
        """Helper to safely get a signature string."""
        try:
            return str(inspect.signature(obj))
        except (ValueError, TypeError):
            return "(...)"
