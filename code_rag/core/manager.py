import logging
import asyncio
import inspect
import os
import shutil
import subprocess  # nosec
import sys
from pathlib import Path
from typing import List, Optional
from .interfaces import IStorage, IParser, IIntelligence
from .models import KnowledgeUnit
from .utils import validate_path
from .constants import MAX_CONCURRENT_TASKS, EMBEDDING_BATCH_SIZE
from .exceptions import StorageError, IntelligenceError
from ..discovery.manager import DiscoveryManager

logger = logging.getLogger(__name__)


def _report_worker_failure(path: str, exc: BaseException) -> tuple[str, str]:
    logger.error("Worker failed to sync %s: %s", path, exc)
    print(f"Worker failed to sync {path}: {exc}", file=sys.stderr, flush=True)
    return (path, str(exc))


def unit_embedding_text(unit: KnowledgeUnit) -> str:
    return unit.summary or (
        f"{unit.kind.value} {unit.name} {unit.signature or ''} {unit.docstring or ''}"
    )


class CodeRAGManager:
    """
    Orchestrates the RAG workflow: parsing, distillation, and storage.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        storage: IStorage,
        parser: IParser,
        intelligence: IIntelligence,
        max_concurrency: int = MAX_CONCURRENT_TASKS,
        allow_build_execution: bool = False,
    ):
        self.storage = storage
        self.parser = parser
        self.intelligence = intelligence
        self.max_concurrency = max_concurrency
        self.allow_build_execution = allow_build_execution
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.discovery = DiscoveryManager(storage=storage)

    async def sync_dependencies(self, project_path: str) -> None:
        """
        Resolves project dependencies (Maven/Gradle) and caches JAR paths.
        """
        root = Path(project_path)
        pom_xml = root / "pom.xml"
        gradle_files = list(root.glob("build.gradle*"))

        if not pom_xml.exists() and not gradle_files:
            return

        if not self.allow_build_execution:
            logger.warning(
                "Dependency sync is disabled by default because Maven/Gradle "
                "build files are executable repository code. Use the explicit "
                "allow_build_execution option only for trusted projects."
            )
            return

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
        # GHSA-wg5p-8h9p-3mr7: Arbitrary Code Execution vulnerability via repository-controlled gradlew.
        # We must ONLY use the system-installed gradle binary.
        gradle_bin: Optional[str] = shutil.which("gradle")

        if not gradle_bin:
            logger.warning(
                "System 'gradle' executable not found in PATH. Skipping dependency sync to avoid executing untrusted repository wrappers."
            )
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

    async def _reject_dirty_incremental(self) -> None:
        if getattr(self.storage, "embedding_model_dirty", False):
            raise StorageError("Embedding model changed; run rebuild or sync --all")

    async def _ensure_embeddings(self) -> None:
        ensure = getattr(self.storage, "ensure_embeddings_bound", None)
        if callable(ensure):
            result = ensure()
            if inspect.isawaitable(result):
                await result

    async def _reembed_all_units(self) -> None:
        units = await self.storage.list_units()
        await self._embed_and_upsert(units)
        await self.storage.mark_embedding_model_synced()

    async def _embed_and_upsert(self, units: list[KnowledgeUnit]) -> None:
        if not units:
            return
        await self._ensure_embeddings()
        embedder = self.storage.embedder
        for start in range(0, len(units), EMBEDDING_BATCH_SIZE):
            chunk = units[start : start + EMBEDDING_BATCH_SIZE]
            texts = [unit_embedding_text(unit) for unit in chunk]
            vectors = await embedder.aembed(texts)
            if len(vectors) != len(chunk):
                raise IntelligenceError("Embedding count mismatch")
            for unit, vector in zip(chunk, vectors):
                await self.storage.upsert_unit(unit, vector=vector)

    async def sync_file(self, file_path: str, force_distill: bool = False) -> None:
        """
        Processes a single file and syncs it with the storage.
        """
        await self._reject_dirty_incremental()
        current_units = await self.parser.distill_file(file_path)
        pending: list[KnowledgeUnit] = []
        for unit in current_units:
            raw_code = unit.metadata.pop("raw_code", "")
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
                    except Exception as exc:
                        logger.error("Failed to distill %s: %s", unit.name, exc)
                        unit.summary = existing_unit.summary if existing_unit else None
            else:
                unit.summary = existing_unit.summary if existing_unit else None

            has_vec = await self.storage.has_embedding(unit.id)
            skip = (
                existing_unit is not None
                and existing_unit.code_hash == unit.code_hash
                and existing_unit.summary == unit.summary
                and has_vec
            )
            if skip:
                await self.storage.upsert_unit(unit)
            else:
                pending.append(unit)
        await self._embed_and_upsert(pending)
        await self.storage.delete_stale_units(
            file_path, [unit.id for unit in current_units]
        )

    async def search(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        """
        Performs semantic search across all indexed units.
        """
        return await self.storage.search_units(query, limit=limit)

    async def sync_project(
        self,
        paths: List[str],
        force_distill: bool = False,
        *,
        index_all: bool = False,
    ) -> list[tuple[str, str]]:
        """
        Concurrent synchronization of multiple files using a worker pool.

        Returns (path, error) pairs for files that failed. The queue is drained
        even if some workers fail; callers must not treat an empty return as
        "raised" — check the list instead of catching after completion.
        """
        if getattr(self.storage, "embedding_model_dirty", False):
            if not index_all:
                raise StorageError("Embedding model changed; run rebuild or sync --all")
            await self._reembed_all_units()
        if not paths:
            return []

        queue: asyncio.Queue[str] = asyncio.Queue()
        for path in paths:
            await queue.put(path)

        async def worker() -> list[tuple[str, str]]:
            failures: list[tuple[str, str]] = []
            while not queue.empty():
                path = await queue.get()
                try:
                    await self.sync_file(
                        path,
                        force_distill=force_distill,
                    )
                except Exception as exc:
                    failures.append(_report_worker_failure(path, exc))
                finally:
                    queue.task_done()
            return failures

        worker_count = min(len(paths), self.max_concurrency)
        tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
        batches = await asyncio.gather(*tasks)
        logger.info("Project sync complete.")
        return [item for batch in batches for item in batch]

    async def close(self) -> None:
        """Closes storage only. Shared embedder lifetime is owned by CodeRAG."""
        await self.storage.close()
        logger.info("CodeRAG manager closed.")
