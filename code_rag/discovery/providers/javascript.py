import json
import logging
from pathlib import Path
from typing import List, Optional

from .base import IDiscoveryProvider
from ...parsers.tree_sitter import TreeSitterParser

logger = logging.getLogger(__name__)


class JavaScriptDiscoveryProvider(IDiscoveryProvider):
    """API discovery for JS/TS libraries using .d.ts files and Tree-Sitter."""

    def __init__(self, parser: Optional[TreeSitterParser] = None):
        self.parser = parser or TreeSitterParser()

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts API using:
        1. Local node_modules/<lib>/package.json -> types/typings
        2. DefinitelyTyped (@types/<lib>)
        3. Static fallback to main .js file
        """
        pkg_root = self._find_package_root(library_name)
        if not pkg_root:
            # Try @types fallback
            pkg_root = self._find_package_root(f"@types/{library_name}")

        if not pkg_root:
            return f"Error: Could not find package '{library_name}' in node_modules."

        # 1. Try to find type definitions (.d.ts)
        target_files = self._get_target_files(pkg_root)

        if not target_files:
            return (
                f"Error: Could not find entry points for '{library_name}' in {pkg_root}"
            )

        output = [f"# Public API for JavaScript/TypeScript Library '{library_name}':"]

        for file_path in target_files[:5]:  # Max 5 files
            try:
                units = await self.parser.distill_file(str(file_path))
                if units:
                    output.append(f"\n## From {file_path.name}:")
                    for unit in units:
                        # For .d.ts, almost everything is interesting
                        output.append(
                            f"- **{unit.kind.value.capitalize()}: {unit.name}**"
                        )
                        output.append(f"  `{unit.signature}`")
            except Exception as e:
                logger.warning("Failed to parse %s: %s", file_path, e)

        return "\n".join(output[:100])

    def _find_package_root(self, lib_name: str) -> Optional[Path]:
        """Locates node_modules folder starting from current dir upwards."""
        curr = Path(".").resolve()
        # Search up to 5 levels up
        for _ in range(5):
            node_modules = curr / "node_modules"
            if node_modules.exists():
                pkg_dir = node_modules / lib_name
                if pkg_dir.exists():
                    return pkg_dir
            if curr.parent == curr:
                break
            curr = curr.parent
        return None

    def _get_target_files(self, pkg_root: Path) -> List[Path]:
        """Identifies .d.ts or .js entry points from package.json."""
        pkg_json_path = pkg_root / "package.json"
        if not pkg_json_path.exists():
            return list(pkg_root.glob("index.d.ts")) or list(pkg_root.glob("index.js"))

        try:
            with open(pkg_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            targets = []
            # 1. Types are best
            type_fields = ["types", "typings"]
            for field in type_fields:
                if field in data:
                    t_path = pkg_root / data[field]
                    if t_path.exists():
                        targets.append(t_path)

            # 2. Exports might have types
            if "exports" in data and isinstance(data["exports"], dict):
                # Simplified check for '.' export
                root_export = data["exports"].get(".")
                if isinstance(root_export, dict) and "types" in root_export:
                    t_path = pkg_root / root_export["types"]
                    if t_path.exists() and t_path not in targets:
                        targets.append(t_path)

            # 3. Main/Module fallback if no types found
            if not targets:
                main_fields = ["module", "main"]
                for field in main_fields:
                    if field in data:
                        m_path = pkg_root / data[field]
                        if m_path.exists():
                            targets.append(m_path)

            return targets
        except Exception as e:
            logger.debug("Failed to read package.json in %s: %s", pkg_root, e)
            return []
