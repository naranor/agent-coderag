import logging
from pathlib import Path
from typing import Optional, Set

from .base import IDiscoveryProvider
from ...parsers.tree_sitter import TreeSitterParser
from ...core.utils import find_directory_upwards

logger = logging.getLogger(__name__)


class CSharpDiscoveryProvider(IDiscoveryProvider):
    """API discovery for C# with Solution and Project traversal."""

    def __init__(self, parser: Optional[TreeSitterParser] = None):
        self.parser = parser or TreeSitterParser()

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts C# API by identifying .sln or .csproj.
        """
        # 1. Find Solution or Project upwards
        target = find_directory_upwards(Path("."), "*.sln") or find_directory_upwards(
            Path("."), "*.csproj"
        )

        if not target:
            return f"Error: Could not find Solution (*.sln) or Project (*.csproj) for '{library_name}'."

        root = target.parent
        output = [f"# Public API for CSharp Project '{library_name}':"]

        # 2. Identify CS files in project
        cs_files = list(root.rglob("*.cs"))

        # Simple recursive static analysis of public entities
        visited_files: Set[Path] = set()
        for file_path in cs_files[:10]:  # Limit scope
            if (
                file_path in visited_files
                or "obj/" in str(file_path)
                or "bin/" in str(file_path)
            ):
                continue

            visited_files.add(file_path)
            try:
                units = await self.parser.distill_file(str(file_path))
                # C# public entities usually start with 'public'
                public_units = [
                    u for u in units if u.signature and "public" in u.signature
                ]

                if public_units:
                    output.append(f"\n## From {file_path.name}:")
                    for u in public_units[:10]:
                        output.append(f"- **{u.kind.value.capitalize()}: {u.name}**")
                        if u.docstring:
                            output.append(f"  *({u.docstring.splitlines()[0]})*")
                        output.append(f"  `{u.signature}`")
            except Exception as e:
                logger.debug("Failed to parse %s: %s", file_path, e)

        return "\n".join(output[:150])
