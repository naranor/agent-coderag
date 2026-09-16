"""Thread-affine DuckDB connection open: RO/RW modes with busy-wait retry.

A dedicated single-thread executor is created per connection and used for
every DuckDB operation (connect, execute, close), since a DuckDB connection
is not safe to use concurrently across threads.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import duckdb

from ..core.exceptions import StorageBusyError, StorageError
from ..core.interfaces import IEmbedder
from .duckdb_impl import AccessMode, DuckDBStorage, apply_rw_schema, finalize_rw_open

__all__ = ["AccessMode", "open_db_connection"]

_BUSY_MARKERS = (
    "could not set lock",
    "conflicting lock",
    "lock on file",
    "database is locked",
    "resource temporarily unavailable",
)

_RETRY_INTERVAL_SECONDS = 0.02


def _is_busy_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _BUSY_MARKERS)


async def _connect_with_retry(
    executor: ThreadPoolExecutor,
    path: str,
    *,
    read_only: bool,
    connect_timeout_seconds: float,
):
    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + connect_timeout_seconds
    while True:
        try:
            return await loop.run_in_executor(
                executor, lambda: duckdb.connect(path, read_only=read_only)
            )
        except Exception as exc:
            if not _is_busy_error(exc):
                raise StorageError(f"Failed to open storage: {exc}") from exc
            if time.monotonic() >= deadline:
                raise StorageBusyError() from exc
            await asyncio.sleep(_RETRY_INTERVAL_SECONDS)


async def open_db_connection(
    path: Path,
    embedder: Optional[IEmbedder],
    *,
    mode: AccessMode,
    connect_timeout_seconds: float = 5.0,
    wipe: bool = False,
) -> DuckDBStorage:
    """Open a thread-affine DuckDB connection in read-only or read-write mode.

    ``embedder`` may be ``None`` only for read-only, metadata-only access
    (e.g. ``get_dependency_path``); vector operations still require it.
    """
    path = Path(path)
    read_only = mode is AccessMode.READ_ONLY

    if read_only:
        if not path.exists():
            raise StorageError(f"Database file not found: {path}")
    elif embedder is None:
        raise StorageError("embedder is required")

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        conn = await _connect_with_retry(
            executor,
            str(path),
            read_only=read_only,
            connect_timeout_seconds=connect_timeout_seconds,
        )
    except Exception:
        executor.shutdown(wait=False)
        raise

    try:
        storage = DuckDBStorage(
            conn, embedder, db_path=str(path), mode=mode, executor=executor
        )
        if not read_only:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                executor, lambda: apply_rw_schema(conn, wipe=wipe)
            )
            finalize_rw_open(storage, wipe=wipe)
        return storage
    except Exception:
        try:
            conn.close()
        except Exception:  # nosec B110
            pass
        executor.shutdown(wait=False)
        raise
