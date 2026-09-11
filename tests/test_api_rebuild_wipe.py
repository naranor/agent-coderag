import duckdb
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.api.client import CodeRAG
from code_rag.api.models import SyncResult
from code_rag.core.manager import CodeRAGManager
from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.storage.duckdb_impl import DuckDBStorage
from tests.embedder_stubs import StubEmbedder


@pytest.mark.asyncio
async def test_rebuild_closes_cached_manager_and_opens_wipe():
    first = MagicMock()
    first.close = AsyncMock()
    second = MagicMock()
    second.close = AsyncMock()
    create = AsyncMock(side_effect=[second])
    with patch("code_rag.api.client.create_manager", new=create), patch(
        "code_rag.api.client.run_rebuild",
        new=AsyncMock(return_value=SyncResult(status="success", indexed_files=1)),
    ) as mock_rebuild:
        rag = CodeRAG(db="proj.db")
        rag._manager = first
        await rag.rebuild()
    first.close.assert_awaited()
    assert create.await_args.kwargs.get("wipe") is True or (
        len(create.await_args.args) >= 1
        and create.call_args_list[-1].kwargs.get("wipe") is True
    )
    mock_rebuild.assert_awaited_once()
    assert mock_rebuild.await_args.args[0] is second
    assert rag._manager is second


@pytest.mark.asyncio
async def test_config_forwards_embedding_kwargs():
    rag = CodeRAG()
    with patch(
        "code_rag.api.client.load_or_update_config", return_value=MagicMock()
    ) as mock_load:
        await rag.config(
            embedding_url="http://e",
            embedding_model="m",
            embedding_key="k",
            embedding_provider="openai",
            clear_embedding=False,
        )
    mock_load.assert_called_once()
    kwargs = mock_load.call_args.kwargs
    assert kwargs["embedding_url"] == "http://e"
    assert kwargs["embedding_model"] == "m"
    assert kwargs["clear_embedding"] is False


@pytest.mark.asyncio
async def test_search_awaits_create_manager():
    manager = MagicMock()
    manager.search = AsyncMock(return_value=[])
    manager.close = AsyncMock()
    with patch(
        "code_rag.api.client.create_manager", new=AsyncMock(return_value=manager)
    ):
        async with CodeRAG() as rag:
            await rag.search("q", limit=3)
    manager.search.assert_awaited_once_with("q", limit=3)


@pytest.mark.asyncio
async def test_live_rebuild_wipes_embeddings_table(tmp_path):
    db = str(tmp_path / "p.db")

    async def factory(db_path, onnx=None, allow_build_execution=False, wipe=False):
        if wipe:
            embedder = StubEmbedder(dim=8, model_id="narrow")
        else:
            embedder = StubEmbedder(dim=16, model_id="wide")
        storage = await DuckDBStorage.open(db_path, embedder, wipe=wipe)
        parser = MagicMock()
        parser.distill_file = AsyncMock(
            return_value=[
                KnowledgeUnit(
                    id="u1",
                    kind=UnitKind.FUNCTION,
                    name="n",
                    path="f.py",
                    code_hash="h",
                    metadata={"raw_code": "def n(): pass"},
                )
            ]
        )
        distiller = MagicMock()
        distiller.summarize = AsyncMock(return_value="s")
        distiller.close = MagicMock()
        return CodeRAGManager(storage, parser, distiller)

    with patch("code_rag.api.client.create_manager", new=factory):
        rag = CodeRAG(db=db, root=tmp_path)
        (tmp_path / "f.py").write_text("def n(): pass\n", encoding="utf-8")
        await rag.sync(path=str(tmp_path / "f.py"))
        wide = duckdb.connect(db)
        typ = [
            r[1]
            for r in wide.execute("DESCRIBE unit_embeddings").fetchall()
            if r[0] == "vec"
        ][0]
        wide.close()
        assert "16" in str(typ)
        await rag.rebuild()
        narrow = duckdb.connect(db)
        typ = [
            r[1]
            for r in narrow.execute("DESCRIBE unit_embeddings").fetchall()
            if r[0] == "vec"
        ][0]
        narrow.close()
        assert "8" in str(typ)
        await rag.close()
