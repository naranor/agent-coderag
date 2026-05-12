import logging
import asyncio
import os
import shutil
import subprocess  # nosec
from pathlib import Path
from typing import List
from .interfaces import IStorage, IParser, IIntelligence
from .models import KnowledgeUnit
from ..discovery.manager import DiscoveryManager

logger = logging.getLogger(__name__)


class CodeRAGManager:
    """
    Orchestrates the RAG workflow: parsing, distillation, and storage.
    """

    def __init__(
        self,
        storage: IStorage,
        parser: IParser,
        intelligence: IIntelligence,
        max_concurrency: int = 10,
    ):
        self.storage = storage
        self.parser = parser
        self.intelligence = intelligence
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.discovery = DiscoveryManager(storage=storage)

    async def sync_dependencies(self, project_path: str):
        """
        Resolves project dependencies (Maven/Gradle) and caches JAR paths.
        """
        root = Path(project_path)
        pom_xml = root / "pom.xml"
        gradle_files = list(root.glob("build.gradle*"))

        if pom_xml.exists():
            await self._sync_maven(root)
        elif gradle_files:
            await self._sync_gradle()

    async def _sync_maven(self, root: Path):
        mvn_bin = shutil.which("mvn")
        if not mvn_bin:
            logger.warning("mvn not found, skipping dependency sync")
            return

        cp_file = root / ".coderag_cp.txt"
        logger.info("Resolving Maven dependencies...")
        try:
            # -Dmdep.outputFile is relative to project root or absolute
            cmd = [
                mvn_bin,
                "dependency:build-classpath",
                f"-Dmdep.outputFile={cp_file.name}",
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )  # nosec
            _, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error("mvn failed: %s", stderr.decode())
                return

            if cp_file.exists():
                with open(cp_file, "r", encoding="utf-8") as f:
                    classpath = f.read().strip()
                cp_file.unlink()

                for jar_path in classpath.split(os.pathsep):
                    if not jar_path:
                        continue
                    # Extract lib name from jar name (e.g. spring-core-6.1.1.jar -> spring-core)
                    jar_name = Path(jar_path).stem
                    # Naive version stripping: everything after last '-' if it starts with digit
                    parts = jar_name.split("-")
                    if len(parts) > 1 and parts[-1][0].isdigit():
                        lib_name = "-".join(parts[:-1])
                    else:
                        lib_name = jar_name

                    await self.storage.set_dependency_path(lib_name, jar_path)
                logger.info("Maven dependencies cached.")
        except Exception as e:
            logger.error("Failed to sync Maven dependencies: %s", e)

    async def _sync_gradle(self):
        # TODO: Implement Gradle resolution
        logger.warning("Gradle dependency sync not yet implemented")

    async def sync_file(self, file_path: str, force_distill: bool = False):
        """
        Processes a single file and syncs it with the storage.
        """
        # 1. Parse AST to get units
        current_units = await self.parser.distill_file(file_path)

        for unit in current_units:
            # v5.40: Delta-distillation logic
            raw_code = unit.metadata.pop("raw_code", "")

            # 2. Get existing unit to check hash
            existing_unit = await self.storage.get_unit(unit.id)

            should_distill = force_distill
            if not existing_unit:
                should_distill = True
                logger.info("New unit discovered: %s", unit.name)
            elif existing_unit.code_hash != unit.code_hash:
                should_distill = True
                logger.info("Unit %s changed (hash mismatch)", unit.name)
            elif not existing_unit.summary:
                should_distill = True
                logger.info("Summary missing for %s", unit.name)

            if should_distill:
                async with self.semaphore:
                    logger.info(
                        "Distilling summary for %s in %s...", unit.name, unit.path
                    )
                    try:
                        unit.summary = await self.intelligence.summarize(
                            raw_code, unit.name
                        )
                    except Exception as e:
                        logger.error("Failed to distill %s: %s", unit.name, e)
                        # Keep old summary if available, otherwise stay None
                        unit.summary = existing_unit.summary if existing_unit else None
            else:
                # Reuse existing summary if code hasn't changed
                unit.summary = existing_unit.summary if existing_unit else None

            # 3. Save to storage (includes embedding generation)
            await self.storage.upsert_unit(unit)

    async def search(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        """
        Performs semantic search across all indexed units.
        """
        return await self.storage.search_units(query, limit=limit)

    async def sync_project(self, paths: List[str], force_distill: bool = False):
        """
        Concurrent synchronization of multiple files.
        """
        tasks = [self.sync_file(p, force_distill=force_distill) for p in paths]
        await asyncio.gather(*tasks)
