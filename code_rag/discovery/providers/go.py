import logging
from pathlib import Path
from typing import Optional

from .base import IDiscoveryProvider
from ...parsers.tree_sitter import TreeSitterParser
from ...core.utils import find_directory_upwards

logger = logging.getLogger(__name__)


class GoDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Go modules."""

    def __init__(self, parser: Optional[TreeSitterParser] = None):
        self.parser = parser or TreeSitterParser()

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts Go API by finding go.mod and parsing exported symbols.
        """
        mod_path = find_directory_upwards(Path("."), "go.mod")
        if not mod_path:
            return f"Error: Could not find go.mod for module '{library_name}'."

        root = mod_path.parent
        main_go = root / "main.go"

        output = [f"# Public API for Go Module '{library_name}':"]

        target_files = [main_go] + list(root.glob("*.go"))
        seen = set()

        for file_path in target_files:
            if not file_path.exists() or file_path in seen:
                continue
            seen.add(file_path)

            try:
                units = await self.parser.distill_file(str(file_path))
                # Go exports are uppercase
                exported = [u for u in units if u.name[0].isupper()]
                if exported:
                    output.append(f"\n## From {file_path.name}:")
                    for u in exported[:20]:
                        output.append(f"- **{u.kind.value.capitalize()}: {u.name}**")
                        if u.docstring:
                            output.append(f"  *({u.docstring.splitlines()[0]})*")
                        output.append(f"  `{u.signature}`")
            except Exception as e:
                logger.debug("Failed to parse %s: %s", file_path, e)

        return "\n".join(output[:150])
