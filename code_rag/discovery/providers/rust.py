import asyncio
import json
import logging
import shutil
import re
from pathlib import Path
from typing import List, Optional, Set

from .base import IDiscoveryProvider
from ...parsers.tree_sitter import TreeSitterParser
from ...core.utils import find_directory_upwards, validate_path
from ...core.constants import METADATA_FETCH_TIMEOUT
from ...core.models import KnowledgeUnit

logger = logging.getLogger(__name__)


class RustDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Rust crates with recursive module traversal."""

    def __init__(self, parser: Optional[TreeSitterParser] = None):
        self.parser = parser or TreeSitterParser()
        self._cargo_bin = self._find_cargo()

    def _find_cargo(self) -> str:
        """Robustly finds the cargo executable."""
        standard = shutil.which("cargo")
        if standard:
            return standard

        # Fallback to standard installation paths if not in PATH
        home = Path.home()
        common_paths = [
            home / ".cargo" / "bin" / "cargo",
            home / ".cargo" / "bin" / "cargo.exe",
        ]
        for p in common_paths:
            if p.exists():
                return str(p)

        return "cargo"  # Fallback to command name

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts Rust API using:
        1. 'cargo metadata' to find crate source path
        2. Recursive static analysis of src/lib.rs and submodules
        """
        # 1. Find crate source using cargo metadata
        crate_root = await self._find_crate_root(library_name)
        if not crate_root:
            return f"Error: Could not find source for Rust crate '{library_name}'."

        # 2. Identify entry points
        entry_points = [crate_root / "src" / "lib.rs", crate_root / "src" / "main.rs"]
        target_files = [p for p in entry_points if p.exists()]

        if not target_files:
            return f"Error: Could not find entry points for crate '{library_name}' in {crate_root}"

        output = [f"# Public API for Rust Crate '{library_name}':"]
        visited_files: Set[Path] = set()
        all_units: List[KnowledgeUnit] = []

        # Start recursive traversal
        for file_path in target_files:
            await self._recursive_extract(file_path, visited_files, all_units, depth=0)

        if not all_units:
            return f"No public entities found for crate '{library_name}'."

        # Group units by file
        units_by_file = {}
        for unit in all_units:
            f_name = Path(unit.path).name
            if f_name not in units_by_file:
                units_by_file[f_name] = []
            units_by_file[f_name].append(unit)

        for f_name, units in units_by_file.items():
            output.append(f"\n## From {f_name}:")
            for unit in units[:20]:
                if unit.kind == "module":
                    continue
                output.append(f"- **{unit.kind.value.capitalize()}: {unit.name}**")
                if unit.docstring:
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
        """Recursively parses modules."""
        max_depth = 3
        if depth > max_depth or file_path in visited or not file_path.exists():
            return

        visited.add(file_path)
        try:
            units = await self.parser.distill_file(str(file_path))
            # Filter only public entities (usually have 'pub' in signature)
            results.extend(
                [
                    u
                    for u in units
                    if u.kind != "module" and u.signature and "pub" in u.signature
                ]
            )

            # Find submodules to follow
            for unit in units:
                target_path = self._parse_submodule(file_path, unit)
                if target_path:
                    await self._recursive_extract(
                        target_path, visited, results, depth + 1
                    )
        except Exception as e:
            logger.warning("Recursive Rust extraction failed for %s: %s", file_path, e)

    def _parse_submodule(
        self, current_file: Path, unit: KnowledgeUnit
    ) -> Optional[Path]:
        """Extracts and resolves a submodule target path from a module unit."""
        if unit.kind == "module" and unit.metadata.get("node_type") == "mod_item":
            # Check if it's a mod without body: 'pub mod name;'
            raw = unit.metadata.get("raw_code", "")
            if ";" in raw and "{" not in raw:
                # Extract module name
                match = re.search(r"mod\s+([a-zA-Z0-9_]+)", raw)
                if match:
                    mod_name = match.group(1)
                    return self._resolve_mod_path(current_file.parent, mod_name)
        return None

    def _resolve_mod_path(self, base_dir: Path, mod_name: str) -> Optional[Path]:
        """Resolves Rust module file path."""
        # 1. name.rs
        p1 = base_dir / f"{mod_name}.rs"
        if p1.exists():
            return p1
        # 2. name/mod.rs
        p2 = base_dir / mod_name / "mod.rs"
        if p2.exists():
            return p2
        return None

    async def _find_crate_root(self, crate_name: str) -> Optional[Path]:
        """Runs 'cargo metadata' to find the path of a dependency."""
        # Basic validation of crate name
        if not all(c.isalnum() or c in "_-" for c in crate_name):
            logger.warning("Invalid crate name: %s", crate_name)
            return None

        manifest_path = find_directory_upwards(Path("."), "Cargo.toml")
        if not manifest_path:
            logger.debug("Cargo.toml not found in project hierarchy")
            return None

        manifest_dir = validate_path(manifest_path.parent)

        try:
            # We NEED deps to find library paths.
            process = await asyncio.create_subprocess_exec(
                self._cargo_bin,
                "metadata",
                "--format-version",
                "1",
                cwd=str(manifest_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=METADATA_FETCH_TIMEOUT
            )

            if process.returncode != 0:
                logger.debug("cargo metadata failed: %s", stderr.decode())
                return None

            data = json.loads(stdout.decode())
            # Search in 'packages' for the crate_name
            for pkg in data.get("packages", []):
                if pkg.get("name") == crate_name:
                    manifest_path = pkg.get("manifest_path")
                    if manifest_path:
                        return Path(manifest_path).parent

        except Exception as e:
            logger.debug("Error running cargo metadata: %s", e)

        return None
