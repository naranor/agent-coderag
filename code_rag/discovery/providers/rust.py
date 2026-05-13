import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from .base import IDiscoveryProvider
from ...parsers.tree_sitter import TreeSitterParser

logger = logging.getLogger(__name__)


class RustDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Rust crates using cargo metadata and Tree-Sitter."""

    def __init__(self, parser: Optional[TreeSitterParser] = None):
        self.parser = parser or TreeSitterParser()
        self._cargo_bin = self._find_cargo()

    def _find_cargo(self) -> str:
        """Robustly finds the cargo executable."""
        # Try standard paths first
        standard = shutil.which("cargo")
        if standard:
            return standard

        # Try specific user path provided earlier
        user_path = r"c:\Users\igbo0122\.cargo\bin\cargo.exe"
        if os.path.exists(user_path):
            return user_path

        return "cargo"  # Fallback to command name

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts Rust API using:
        1. 'cargo metadata' to find crate source path
        2. Static analysis of src/lib.rs or src/main.rs using Tree-Sitter
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

        for file_path in target_files:
            try:
                units = await self.parser.distill_file(str(file_path))
                if units:
                    output.append(f"\n## From {file_path.name}:")
                    for unit in units:
                        # Only show public entities (Tree-Sitter signatures usually include 'pub')
                        if unit.signature and "pub" in unit.signature:
                            output.append(
                                f"- **{unit.kind.value.capitalize()}: {unit.name}**"
                            )
                            output.append(f"  `{unit.signature}`")
            except Exception as e:
                logger.warning("Failed to parse %s: %s", file_path, e)

        return "\n".join(output[:100])

    async def _find_crate_root(self, crate_name: str) -> Optional[Path]:
        """Runs 'cargo metadata' to find the path of a dependency."""
        manifest_dir = self._find_manifest_dir()
        if not manifest_dir:
            logger.debug("Cargo.toml not found in project hierarchy")
            return None

        try:
            process = await asyncio.create_subprocess_exec(
                self._cargo_bin,
                "metadata",
                "--format-version",
                "1",
                "--no-deps",  # Speed up if we only care about direct workspace, but we usually want deps
                cwd=str(manifest_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Wait, actually we NEED deps to find library paths. Removing --no-deps.
            process = await asyncio.create_subprocess_exec(
                self._cargo_bin,
                "metadata",
                "--format-version",
                "1",
                cwd=str(manifest_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

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

    def _find_manifest_dir(self) -> Optional[Path]:
        """Locates Cargo.toml folder starting from current dir upwards."""
        curr = Path(".").resolve()
        for _ in range(5):
            if (curr / "Cargo.toml").exists():
                return curr
            if curr.parent == curr:
                break
            curr = curr.parent
        return None
