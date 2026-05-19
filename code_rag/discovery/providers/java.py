import logging
import zipfile
import os
from typing import Optional

from .base import IDiscoveryProvider
from ...core.interfaces import IStorage
from ...parsers.tree_sitter import TreeSitterParser

logger = logging.getLogger(__name__)


class JavaDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Java libraries using bytecode/source analysis."""

    def __init__(
        self,
        storage: Optional[IStorage] = None,
        parser: Optional[TreeSitterParser] = None,
    ):
        self.storage = storage
        self.parser = parser or TreeSitterParser()

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts Java API from a cached JAR file.
        """
        if not self.storage:
            return "Error: Storage required for Java API discovery."

        jar_path = await self.storage.get_dependency_path(library_name)
        if not jar_path or not os.path.exists(jar_path):
            return f"Error: Could not find cached JAR for '{library_name}'. Run 'sync' first."

        output = [f"# Public API for Java Library '{library_name}':"]
        try:
            with zipfile.ZipFile(jar_path, "r") as jar:
                # 1. Look for public class names in entry list
                class_files = [
                    f for f in jar.namelist() if f.endswith(".class") and "$" not in f
                ]

                if not class_files:
                    return f"No public classes found in {jar_path}"

                for cf in class_files[:20]:
                    # Convert internal name to human-readable
                    class_name = cf.replace("/", ".").replace(".class", "")
                    output.append(f"- **Class: {class_name}**")

            return "\n".join(output)
        except Exception as e:
            return f"Error reading JAR {jar_path}: {e}"
