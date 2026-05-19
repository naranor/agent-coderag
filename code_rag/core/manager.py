import logging
import asyncio
import os
import shutil
import subprocess  # nosec
from pathlib import Path
from typing import List, Optional
from .interfaces import IStorage, IParser, IIntelligence
from .models import KnowledgeUnit
from .utils import validate_path
from .constants import MAX_CONCURRENT_TASKS
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
        max_concurrency: int = MAX_CONCURRENT_TASKS,
    ):
        self.storage = storage
        self.parser = parser
        self.intelligence = intelligence
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.discovery = DiscoveryManager(storage=storage)

    async def sync_dependencies(self, project_path: str) -> None:
        """
        Resolves project dependencies (Maven/Gradle) and caches JAR paths.
        """
        root = Path(project_path)
        pom_xml = root / "pom.xml"
        gradle_files = list(root.glob("build.gradle*"))

        if pom_xml.exists():
            await self._sync_maven(root)
        elif gradle_files:
            await self._sync_gradle(root)

    async def _sync_maven(self, root: Path) -> None:
        mvn_bin: Optional[str] = shutil.which("mvn")
        if not mvn_bin:
            logger.warning("mvn not found, skipping dependency sync")
            return

        # Validate path to prevent path traversal/injection
        valid_root = validate_path(root)

        cp_file = valid_root / ".coderag_cp.txt"
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
                cwd=str(valid_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )  # nosec
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

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
                    await self._cache_jar_path(jar_path)
                logger.info("Maven dependencies cached.")
        except Exception as e:
            logger.error("Failed to sync Maven dependencies: %s", e)

    async def _cache_jar_path(self, jar_path: str) -> None:
        """Helper to parse jar name and cache it."""
        # Extract lib name from jar name (e.g. spring-core-6.1.1.jar -> spring-core)
        jar_name = Path(jar_path).stem
        # Naive version stripping: everything after last '-' if it starts with digit
        parts = jar_name.split("-")
        if len(parts) > 1 and parts[-1][0].isdigit():
            lib_name = "-".join(parts[:-1])
        else:
            lib_name = jar_name

        await self.storage.set_dependency_path(lib_name, jar_path)

    async def _sync_gradle(self, root: Path) -> None:
        """
        Resolves Gradle dependencies using a temporary init script.
        """
        # 1. Find gradle executable (wrapper preferred)
        gradle_wrapper = root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        gradle_bin: Optional[str] = None
        if gradle_wrapper.exists():
            gradle_bin = str(gradle_wrapper.resolve())
        else:
            gradle_bin = shutil.which("gradle")

        if not gradle_bin:
            logger.warning("gradle not found, skipping dependency sync")
            return

        # Validate path
        valid_root = validate_path(root)
        await self._execute_gradle_init(valid_root, gradle_bin)

    async def _execute_gradle_init(self, root: Path, gradle_bin: str) -> None:
        init_script = root / ".coderag_init.gradle"
        init_content = """
allprojects {
    tasks.register('printCodeRagCP') {
        doLast {
            def cp = []
            ['runtimeClasspath', 'compileClasspath', 'implementation'].each { cfgName ->
                def cfg = project.configurations.findByName(cfgName)
                if (cfg != null && cfg.canBeResolved) {
                    try {
                        cp.addAll(cfg.collect { it.absolutePath })
                    } catch (Exception e) {}
                }
            }
            if (cp) {
                println "CODERAG_CP:" + cp.unique().join(File.pathSeparator)
            }
        }
    }
}
"""
        logger.info("Resolving Gradle dependencies...")
        try:
            with open(init_script, "w", encoding="utf-8") as f:
                f.write(init_content)

            process = await asyncio.create_subprocess_exec(
                gradle_bin,
                "-q",
                "--init-script",
                str(init_script),
                "printCodeRagCP",
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

            if process.returncode != 0:
                logger.error("gradle failed: %s", stderr.decode())
                return

            output = stdout.decode()
            for line in output.splitlines():
                if line.startswith("CODERAG_CP:"):
                    classpath = line[len("CODERAG_CP:") :].strip()
                    for jar_path in classpath.split(os.pathsep):
                        if not jar_path or not jar_path.endswith(".jar"):
                            continue
                        await self._cache_jar_path(jar_path)

            logger.info("Gradle dependencies cached.")
        except Exception as e:
            logger.error("Failed to sync Gradle dependencies: %s", e)
        finally:
            if init_script.exists():
                init_script.unlink()

    async def sync_file(self, file_path: str, force_distill: bool = False) -> None:
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
                        summary = await self.intelligence.summarize(raw_code, unit.name)
                        unit.summary = summary
                    except Exception as e:
                        logger.error("Failed to distill %s: %s", unit.name, e)
                        # Keep old summary if available, otherwise stay None
                        unit.summary = existing_unit.summary if existing_unit else None
            else:
                # Reuse existing summary if code hasn't changed
                unit.summary = existing_unit.summary if existing_unit else None

            # 3. Save to storage (includes embedding generation)
            await self.storage.upsert_unit(unit)

        # 4. Garbage Collection: Remove units that were in this file but are no longer there
        current_unit_ids = [u.id for u in current_units]
        await self.storage.delete_stale_units(file_path, current_unit_ids)

    async def search(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        """
        Performs semantic search across all indexed units.
        """
        return await self.storage.search_units(query, limit=limit)

    async def sync_project(self, paths: List[str], force_distill: bool = False) -> None:
        """
        Concurrent synchronization of multiple files using a worker pool.
        """
        if not paths:
            return

        queue: asyncio.Queue[str] = asyncio.Queue()
        for p in paths:
            await queue.put(p)

        async def worker() -> None:
            while not queue.empty():
                path = await queue.get()
                try:
                    await self.sync_file(path, force_distill=force_distill)
                except Exception as e:
                    logger.error("Worker failed to sync %s: %s", path, e)
                finally:
                    queue.task_done()

        # Run limited number of workers
        worker_count = min(len(paths), self.max_concurrency)
        tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]

        await asyncio.gather(*tasks)
        logger.info("Project sync complete.")

    async def close(self) -> None:
        """Releases manager resources."""
        await self.storage.close()
        # If intelligence has close method (Embedder does)
        if hasattr(self.intelligence, "close"):
            self.intelligence.close()
        logger.info("CodeRAG manager closed.")
