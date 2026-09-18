"""Indexing use-cases: sync one file or a project tree into an open storage."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional

from code_rag.core.constants import EMBEDDING_BATCH_SIZE, MAX_CONCURRENT_TASKS
from code_rag.core.exceptions import IntelligenceError, StorageError
from code_rag.core.interfaces import IIntelligence, IParser, IStorage
from code_rag.core.models import KnowledgeUnit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexStack:
    """Ports for indexing use-cases (no resource ownership)."""

    storage: IStorage
    parser: IParser
    intelligence: IIntelligence


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


def _should_distill_unit(
    existing: Optional[KnowledgeUnit], unit: KnowledgeUnit, force: bool
) -> bool:
    if force or existing is None:
        if existing is None:
            logger.info("New unit discovered: %s", unit.name)
        return True
    if existing.code_hash != unit.code_hash:
        logger.info("Unit %s changed (hash mismatch)", unit.name)
        return True
    if not existing.summary:
        logger.info("Summary missing for %s", unit.name)
        return True
    return False


async def _distill_unit_summary(
    intelligence: IIntelligence,
    unit: KnowledgeUnit,
    existing: Optional[KnowledgeUnit],
    raw_code: str,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        logger.info("Distilling summary for %s in %s...", unit.name, unit.path)
        try:
            unit.summary = await intelligence.summarize(raw_code, unit.name)
        except Exception as exc:
            logger.error("Failed to distill %s: %s", unit.name, exc)
            unit.summary = existing.summary if existing else None


async def _unit_needs_embedding(
    storage: IStorage, unit: KnowledgeUnit, existing: Optional[KnowledgeUnit]
) -> bool:
    has_vec = await storage.has_embedding(unit.id)
    if existing is None:
        return True
    return not (
        existing.code_hash == unit.code_hash
        and existing.summary == unit.summary
        and has_vec
    )


async def _process_unit(
    stack: IndexStack,
    unit: KnowledgeUnit,
    *,
    force_distill: bool,
    semaphore: asyncio.Semaphore,
    pending: list[KnowledgeUnit],
) -> None:
    raw_code = unit.metadata.pop("raw_code", "")
    existing = await stack.storage.get_unit(unit.id)
    if _should_distill_unit(existing, unit, force_distill):
        await _distill_unit_summary(
            stack.intelligence, unit, existing, raw_code, semaphore
        )
    else:
        unit.summary = existing.summary if existing else None

    if await _unit_needs_embedding(stack.storage, unit, existing):
        pending.append(unit)
    else:
        await stack.storage.upsert_unit(unit)


async def sync_file(
    stack: IndexStack,
    file_path: str,
    *,
    force_distill: bool = False,
    max_concurrency: int = MAX_CONCURRENT_TASKS,
) -> None:
    """Parse, distill, embed, and upsert one file into ``stack.storage``."""
    await _reject_dirty_incremental(stack.storage)
    semaphore = asyncio.Semaphore(max_concurrency)
    current_units = await stack.parser.distill_file(file_path)
    pending: list[KnowledgeUnit] = []
    for unit in current_units:
        await _process_unit(
            stack,
            unit,
            force_distill=force_distill,
            semaphore=semaphore,
            pending=pending,
        )
    await _embed_and_upsert(stack.storage, pending)
    await stack.storage.delete_stale_units(
        file_path, [unit.id for unit in current_units]
    )


async def sync_project(
    stack: IndexStack,
    paths: List[str],
    *,
    force_distill: bool = False,
    index_all: bool = False,
    max_concurrency: int = MAX_CONCURRENT_TASKS,
) -> list[tuple[str, str]]:
    """Concurrent sync of multiple files. Returns (path, error) failure pairs."""
    if getattr(stack.storage, "embedding_model_dirty", False):
        if not index_all:
            raise StorageError("Embedding model changed; run rebuild or sync --all")
        await _reembed_all_units(stack.storage)
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
                    stack,
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
