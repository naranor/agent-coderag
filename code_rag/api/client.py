from pathlib import Path
from typing import Optional, Union

from code_rag.api.models import ApiReport, SetupResult, SyncResult
from code_rag.core.models import KnowledgeUnit
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.services.config import load_or_update_config
from code_rag.services.discovery_api import run_api
from code_rag.services.factory import create_manager
from code_rag.services.search import run_search
from code_rag.services.setup import run_setup
from code_rag.services.sync import run_rebuild, run_sync


class CodeRAG:
    """Public async facade over CodeRAG services and the storage/search manager."""

    def __init__(
        self,
        db: str = "code_rag.db",
        onnx: Optional[str] = None,
        *,
        root: Optional[Union[str, Path]] = None,
        allow_build_execution: bool = False,
    ):
        self._db = db
        self._onnx = onnx
        self._root = Path(root) if root is not None else Path.cwd()
        self._allow_build_execution = allow_build_execution
        self._manager = None

    async def __aenter__(self) -> "CodeRAG":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._manager is not None:
            await self._manager.close()
            self._manager = None

    def _ensure_manager(self):
        if self._manager is None:
            self._manager = create_manager(
                self._db, self._onnx, allow_build_execution=self._allow_build_execution
            )
        return self._manager

    async def config(
        self,
        *,
        url: Optional[str] = None,
        key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> DistillerConfig:
        return load_or_update_config(url=url, key=key, model=model, provider=provider)

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
        return await run_sync(
            self._ensure_manager(),
            root=self._root,
            path=path,
            index_all=index_all,
            force=force,
        )

    async def search(self, query: str, *, limit: int = 5) -> list[KnowledgeUnit]:
        return await run_search(self._ensure_manager(), query, limit=limit)

    async def api(self, library: str, *, lang: Optional[str] = None) -> ApiReport:
        return await run_api(self._ensure_manager(), library, lang=lang)

    async def rebuild(self) -> SyncResult:
        return await run_rebuild(self._ensure_manager(), root=self._root)
