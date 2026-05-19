import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Set, Dict

from .base import IDiscoveryProvider
from ...parsers.tree_sitter import TreeSitterParser
from ...core.utils import find_directory_upwards
from ...core.exceptions import DiscoveryError
from ...core.models import KnowledgeUnit

logger = logging.getLogger(__name__)


class JavaScriptDiscoveryProvider(IDiscoveryProvider):
    """API discovery for JS/TS libraries with recursive export traversal."""

    def __init__(self, parser: Optional[TreeSitterParser] = None):
        self.parser = parser or TreeSitterParser()

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts API using:
        1. Local node_modules/<lib>/package.json -> types/typings
        2. DefinitelyTyped (@types/<lib>)
        3. Static fallback to main .js file
        4. Recursive traversal of local exports
        """
        pkg_root = self._find_package_root(library_name)
        if not pkg_root:
            # Try @types fallback
            pkg_root = self._find_package_root(f"@types/{library_name}")

        if not pkg_root:
            raise DiscoveryError(
                f"Could not find package '{library_name}' in node_modules."
            )

        # 1. Try to find entry points
        target_files = self._get_target_files(pkg_root)

        if not target_files:
            raise DiscoveryError(
                f"Could not find entry points for '{library_name}' in {pkg_root}"
            )

        output = [f"# Public API for JavaScript/TypeScript Library '{library_name}':"]
        visited_files: Set[Path] = set()
        all_units: List[KnowledgeUnit] = []

        # Start recursive traversal from entry points
        for file_path in target_files[:3]:  # Limit entry points
            await self._recursive_extract(file_path, visited_files, all_units, depth=0)

        if not all_units:
            return f"No public entities found for '{library_name}'."

        # Group units by file for display
        units_by_file: Dict[str, List[KnowledgeUnit]] = {}
        for unit in all_units:
            f_name = Path(unit.path).name
            if f_name not in units_by_file:
                units_by_file[f_name] = []
            units_by_file[f_name].append(unit)

        for f_name, units in units_by_file.items():
            output.append(f"\n## From {f_name}:")
            for unit in units[:20]:  # Limit units per file
                if unit.kind == "module":
                    continue
                output.append(f"- **{unit.kind.value.capitalize()}: {unit.name}**")
                if unit.docstring:
                    # Brief docstring (first line)
                    first_line = unit.docstring.splitlines()[0]
                    output.append(f"  *({first_line})*")
                output.append(f"  `{unit.signature}`")

        return "\n".join(output[:150])

    async def _recursive_extract(
        self,
        file_path: Path,
        visited: Set[Path],
        results: List[KnowledgeUnit],
        depth: int,
    ):
        """Recursively parses files following local exports."""
        max_depth = 3
        if depth > max_depth or file_path in visited or not file_path.exists():
            return

        visited.add(file_path)
        try:
            units = await self.parser.distill_file(str(file_path))
            results.extend([u for u in units if u.kind != "module"])

            # Find exports to follow
            for unit in units:
                await self._follow_exports(
                    unit,
                    file_path,
                    {"visited": visited, "results": results, "depth": depth},
                )
        except Exception as e:
            logger.warning("Recursive extraction failed for %s: %s", file_path, e)

    async def _follow_exports(self, unit, file_path, ctx):
        if unit.kind == "module" and "export" in unit.metadata.get("node_type", ""):
            # Try to extract target path from export statement
            # Example: export * from './utils'
            raw = unit.metadata.get("raw_code", "")
            match = re.search(r"from\s+['\"]([^'\"]+)['\"]", raw)
            if match:
                target = match.group(1)
                if target.startswith("."):
                    # Resolve relative path
                    target_path = self._resolve_local_path(file_path.parent, target)
                    if target_path:
                        await self._recursive_extract(
                            target_path,
                            ctx["visited"],
                            ctx["results"],
                            ctx["depth"] + 1,
                        )

    def _resolve_local_path(self, base_dir: Path, target: str) -> Optional[Path]:
        """Resolves local JS/TS import path (handling extensions)."""
        p = (base_dir / target).resolve()
        # Try various extensions
        for ext in (".d.ts", ".ts", ".js", "/index.d.ts", "/index.ts", "/index.js"):
            full_p = (
                p.with_suffix(ext) if not ext.startswith("/") else Path(str(p) + ext)
            )
            if full_p.exists():
                return full_p
        return None

    def _find_package_root(self, lib_name: str) -> Optional[Path]:
        """Locates node_modules folder starting from current dir upwards."""
        nm_path = find_directory_upwards(Path("."), "node_modules")
        if nm_path:
            pkg_dir = nm_path / lib_name
            if pkg_dir.exists():
                return pkg_dir
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
                export_target = self._parse_exports_field(pkg_root, data["exports"])
                if export_target and export_target not in targets:
                    targets.append(export_target)

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

    def _parse_exports_field(
        self, pkg_root: Path, exports_data: dict
    ) -> Optional[Path]:
        """Helper to parse the exports field for types."""
        root_export = exports_data.get(".")
        if isinstance(root_export, dict) and "types" in root_export:
            t_path = pkg_root / root_export["types"]
            if t_path.exists():
                return t_path
        return None
