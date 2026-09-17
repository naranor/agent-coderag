import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional, Union

from code_rag.api.models import ApiReport, SetupResult, SyncResult
from code_rag.core.constants import DEFAULT_CONNECT_TIMEOUT_SECONDS
from code_rag.core.interfaces import IEmbedder, IIntelligence, IParser
from code_rag.core.models import KnowledgeUnit
from code_rag.discovery.manager import DiscoveryManager
from code_rag.discovery.providers.java import JavaDiscoveryProvider
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.paths import resolve_db_path
from code_rag.services.config import load_or_update_config
from code_rag.services.discovery_api import run_api
from code_rag.services.factory import create_stack
from code_rag.services.search import run_search
from code_rag.services.setup import run_setup
from code_rag.services.indexing import IndexStack
from code_rag.services.sync import SyncOptions, run_rebuild, run_sync
from code_rag.storage.db_connection import AccessMode, open_db_connection
from code_rag.storage.duckdb_impl import DuckDBStorage

logger = logging.getLogger(__name__)


class CodeRAG:  # pylint: disable=too-many-instance-attributes
    """Public async facade: process-scoped stack + ephemeral DB per operation."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        db: Optional[str] = None,
        onnx: Optional[str] = None,
        *,
        root: Optional[Union[str, Path]] = None,
        allow_build_execution: bool = False,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ):
        self._onnx = onnx
        self._root = Path(root) if root is not None else Path.cwd()
        self._allow_build_execution = allow_build_execution
        self._connect_timeout_seconds = connect_timeout_seconds
        self._db_path = resolve_db_path(db, root=self._root)
        self._embedder: Optional[IEmbedder] = None
        self._parser: Optional[IParser] = None
        self._distiller: Optional[IIntelligence] = None
        self._storage: Optional[DuckDBStorage] = None
        self._op_lock = asyncio.Lock()
        logger.info("Resolved database path: %s", self._db_path.resolve())

    async def __aenter__(self) -> "CodeRAG":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def _ensure_stack(self) -> None:
        if self._embedder is not None:
            return
        self._embedder, self._parser, self._distiller = await create_stack(self._onnx)

    @asynccontextmanager
    async def _with_storage(
        self, mode: AccessMode, *, wipe: bool = False
    ) -> AsyncIterator[DuckDBStorage]:
        async with self._op_lock:
            await self._ensure_stack()
            if self._embedder is None:
                raise RuntimeError("process stack is not initialized")
            storage = await open_db_connection(
                self._db_path,
                self._embedder,
                mode=mode,
                connect_timeout_seconds=self._connect_timeout_seconds,
                wipe=wipe,
            )
            self._storage = storage
            try:
                yield storage
            finally:
                await storage.close()
                self._storage = None

    @asynccontextmanager
    async def _metadata_ro_connection(self) -> AsyncIterator[DuckDBStorage]:
        storage = await open_db_connection(
            self._db_path,
            None,
            mode=AccessMode.READ_ONLY,
            connect_timeout_seconds=self._connect_timeout_seconds,
        )
        self._storage = storage
        try:
            yield storage
        finally:
            await storage.close()
            self._storage = None

    async def close(self) -> None:
        async with self._op_lock:
            if self._storage is not None:
                await self._storage.close()
                self._storage = None
            if self._embedder is not None:
                await self._embedder.close()
                self._embedder = None
            self._parser = None
            self._distiller = None

    async def config(  # pylint: disable=too-many-arguments
        self,
        *,
        url: Optional[str] = None,
        key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        embedding_url: Optional[str] = None,
        embedding_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        clear_embedding: bool = False,
    ) -> DistillerConfig:
        return load_or_update_config(
            url=url,
            key=key,
            model=model,
            provider=provider,
            embedding_url=embedding_url,
            embedding_key=embedding_key,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            clear_embedding=clear_embedding,
        )

    async def setup(self, *, force: bool = False) -> SetupResult:
        return await run_setup(force=force)

    async def sync(
        self,
        path: Optional[str] = None,
        *,
        index_all: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if path is None and not index_all:
            return SyncResult(status="success", indexed_files=0)
        async with self._with_storage(AccessMode.READ_WRITE) as storage:
            if self._parser is None or self._distiller is None:
                raise RuntimeError("process stack is not initialized")
            stack = IndexStack(storage, self._parser, self._distiller)
            return await run_sync(
                stack,
                SyncOptions(
                    root=self._root,
                    path=path,
                    index_all=index_all,
                    force=force,
                    allow_build_execution=self._allow_build_execution,
                ),
            )

    async def search(self, query: str, *, limit: int = 5) -> list[KnowledgeUnit]:
        async with self._with_storage(AccessMode.READ_ONLY) as storage:
            return await run_search(storage, query, limit=limit)

    async def _api_java(
        self,
        discovery: DiscoveryManager,
        provider: JavaDiscoveryProvider,
        library: str,
        language: str,
    ) -> ApiReport:
        async with self._op_lock:
            async with self._metadata_ro_connection() as storage:
                provider.storage = storage
                try:
                    return await run_api(discovery, library, lang=language)
                finally:
                    provider.storage = None

    async def api(self, library: str, *, lang: Optional[str] = None) -> ApiReport:
        language = lang or "python"
        discovery = DiscoveryManager()
        provider = discovery.get_provider(language)
        if isinstance(provider, JavaDiscoveryProvider):
            return await self._api_java(discovery, provider, library, language)
        return await run_api(discovery, library, lang=language)

    async def rebuild(self) -> SyncResult:
        async with self._with_storage(AccessMode.READ_WRITE, wipe=True) as storage:
            if self._parser is None or self._distiller is None:
                raise RuntimeError("process stack is not initialized")
            stack = IndexStack(storage, self._parser, self._distiller)
            return await run_rebuild(
                stack,
                root=self._root,
                allow_build_execution=self._allow_build_execution,
            )
