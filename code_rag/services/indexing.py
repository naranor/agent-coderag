"""Indexing use-cases: sync one file or a project tree into an open storage."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from typing import List

from code_rag.core.constants import EMBEDDING_BATCH_SIZE, MAX_CONCURRENT_TASKS
from code_rag.core.exceptions import IntelligenceError, StorageError
from code_rag.core.interfaces import IIntelligence, IParser, IStorage
from code_rag.core.models import KnowledgeUnit

logger = logging.getLogger(__name__)


def _report_worker_failure(path: str, exc: BaseException) -> tuple[str, str]:
    logger.error("Worker failed to sync %s: %s", path, exc)
    print(f"Worker failed to sync {path}: {exc}", file=sys.stderr, flush=True)
    return (path, str(exc))


def unit_embedding_text(unit: KnowledgeUnit) -> str:
    return unit.summary or (
        f"{unit.kind.value} {unit.name} {unit.signature or ''} {unit.docstring or ''}"
    )


async def _ensure_embeddings(storage: IStorage) -> None:
    ensure = getattr(storage, "ensure_embeddings_bound", None)
    if callable(ensure):
        result = ensure()
        if inspect.isawaitable(result):
            await result


async def _embed_and_upsert(storage: IStorage, units: list[KnowledgeUnit]) -> None:
    if not units:
        return
    await _ensure_embeddings(storage)
    embedder = storage.embedder
    for start in range(0, len(units), EMBEDDING_BATCH_SIZE):
        chunk = units[start : start + EMBEDDING_BATCH_SIZE]
        texts = [unit_embedding_text(unit) for unit in chunk]
        vectors = await embedder.aembed(texts)
        if len(vectors) != len(chunk):
            raise IntelligenceError("Embedding count mismatch")
        for unit, vector in zip(chunk, vectors):
            await storage.upsert_unit(unit, vector=vector)


async def _reembed_all_units(storage: IStorage) -> None:
    units = await storage.list_units()
    await _embed_and_upsert(storage, units)
    await storage.mark_embedding_model_synced()


async def _reject_dirty_incremental(storage: IStorage) -> None:
    if getattr(storage, "embedding_model_dirty", False):
        raise StorageError("Embedding model changed; run rebuild or sync --all")


async def sync_file(  # pylint: disable=too-many-arguments,too-many-locals
    storage: IStorage,
    parser: IParser,
    intelligence: IIntelligence,
    file_path: str,
    *,
    force_distill: bool = False,
    max_concurrency: int = MAX_CONCURRENT_TASKS,
) -> None:
    """Parse, distill, embed, and upsert one file into ``storage``."""
    await _reject_dirty_incremental(storage)
    semaphore = asyncio.Semaphore(max_concurrency)
    current_units = await parser.distill_file(file_path)
    pending: list[KnowledgeUnit] = []
    for unit in current_units:
        raw_code = unit.metadata.pop("raw_code", "")
        existing_unit = await storage.get_unit(unit.id)
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
            async with semaphore:
                logger.info("Distilling summary for %s in %s...", unit.name, unit.path)
                try:
                    unit.summary = await intelligence.summarize(raw_code, unit.name)
                except Exception as exc:
                    logger.error("Failed to distill %s: %s", unit.name, exc)
                    unit.summary = existing_unit.summary if existing_unit else None
        else:
            unit.summary = existing_unit.summary if existing_unit else None

        has_vec = await storage.has_embedding(unit.id)
        skip = (
            existing_unit is not None
            and existing_unit.code_hash == unit.code_hash
            and existing_unit.summary == unit.summary
            and has_vec
        )
        if skip:
            await storage.upsert_unit(unit)
        else:
            pending.append(unit)
    await _embed_and_upsert(storage, pending)
    await storage.delete_stale_units(file_path, [unit.id for unit in current_units])


async def sync_project(  # pylint: disable=too-many-arguments
    storage: IStorage,
    parser: IParser,
    intelligence: IIntelligence,
    paths: List[str],
    *,
    force_distill: bool = False,
    index_all: bool = False,
    max_concurrency: int = MAX_CONCURRENT_TASKS,
) -> list[tuple[str, str]]:
    """Concurrent sync of multiple files. Returns (path, error) failure pairs."""
    if getattr(storage, "embedding_model_dirty", False):
        if not index_all:
            raise StorageError("Embedding model changed; run rebuild or sync --all")
        await _reembed_all_units(storage)
    if not paths:
        return []

    queue: asyncio.Queue[str] = asyncio.Queue()
    for path in paths:
        await queue.put(path)

    async def worker() -> list[tuple[str, str]]:
        failures: list[tuple[str, str]] = []
        while True:
            try:
                path = queue.get_nowait()
            except asyncio.QueueEmpty:
                return failures
            try:
                await sync_file(
                    storage,
                    parser,
                    intelligence,
                    path,
                    force_distill=force_distill,
                    max_concurrency=max_concurrency,
                )
            except Exception as exc:
                failures.append(_report_worker_failure(path, exc))
            finally:
                queue.task_done()

    worker_count = min(len(paths), max_concurrency)
    tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
    batches = await asyncio.gather(*tasks)
    logger.info("Project sync complete.")
    return [item for batch in batches for item in batch]
