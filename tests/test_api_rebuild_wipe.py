import duckdb
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.api.client import CodeRAG
from code_rag.api.models import ApiReport, SyncResult
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.storage.db_connection import AccessMode
from tests.embedder_stubs import StubEmbedder


@pytest.mark.asyncio
async def test_rebuild_opens_once_with_wipe(tmp_path, monkeypatch):
    opens: list[dict] = []
    embedder = StubEmbedder(dim=8)
    storage = MagicMock()
    storage.close = AsyncMock()

    async def tracking_open(
        path, emb, *, mode, connect_timeout_seconds=5.0, wipe=False
    ):
        opens.append({"mode": mode, "wipe": wipe, "emb": emb})
        return storage

    monkeypatch.setattr(
        "code_rag.api.client.create_stack",
        AsyncMock(return_value=(embedder, MagicMock(), MagicMock())),
    )
    monkeypatch.setattr("code_rag.api.client.open_db_connection", tracking_open)
    monkeypatch.setattr(
        "code_rag.api.client.build_manager", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        "code_rag.api.client.run_rebuild",
        AsyncMock(return_value=SyncResult(status="success", indexed_files=1)),
    )

    rag = CodeRAG(db=str(tmp_path / "proj.db"), root=tmp_path)
    try:
        await rag.rebuild()
    finally:
        await rag.close()

    assert len(opens) == 1
    assert opens[0]["wipe"] is True
    assert opens[0]["mode"] is AccessMode.READ_WRITE
    assert opens[0]["emb"] is embedder
    storage.close.assert_awaited()


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
async def test_search_uses_read_only_connection(tmp_path, monkeypatch):
    opens: list[AccessMode] = []
    storage = MagicMock()
    storage.close = AsyncMock()
    embedder = StubEmbedder(dim=8)

    async def tracking_open(
        path, emb, *, mode, connect_timeout_seconds=5.0, wipe=False
    ):
        opens.append(mode)
        return storage

    monkeypatch.setattr(
        "code_rag.api.client.create_stack",
        AsyncMock(return_value=(embedder, MagicMock(), MagicMock())),
    )
    monkeypatch.setattr("code_rag.api.client.open_db_connection", tracking_open)
    monkeypatch.setattr(
        "code_rag.api.client.build_manager", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr("code_rag.api.client.run_search", AsyncMock(return_value=[]))

    async with CodeRAG(db=str(tmp_path / "p.db"), root=tmp_path) as rag:
        await rag.search("q", limit=3)

    assert opens == [AccessMode.READ_ONLY]


@pytest.mark.asyncio
async def test_live_rebuild_wipes_embeddings_table(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    (tmp_path / "f.py").write_text("def n(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        "code_rag.intelligence.distiller.Distiller.summarize",
        AsyncMock(return_value="s"),
    )
    monkeypatch.setattr(
        "code_rag.services.factory.DistillerConfig.load",
        lambda: DistillerConfig(),
    )

    wide = StubEmbedder(dim=16, model_id="wide")
    monkeypatch.setattr(
        "code_rag.services.factory.create_embedder",
        AsyncMock(return_value=wide),
    )
    rag = CodeRAG(db=db, root=tmp_path)
    try:
        await rag.sync(path=str(tmp_path / "f.py"))
    finally:
        await rag.close()

    wide_conn = duckdb.connect(db)
    typ = [
        row[1]
        for row in wide_conn.execute("DESCRIBE unit_embeddings").fetchall()
        if row[0] == "vec"
    ][0]
    wide_conn.close()
    assert "16" in str(typ)

    narrow = StubEmbedder(dim=8, model_id="narrow")
    monkeypatch.setattr(
        "code_rag.services.factory.create_embedder",
        AsyncMock(return_value=narrow),
    )
    rag2 = CodeRAG(db=db, root=tmp_path)
    try:
        await rag2.rebuild()
    finally:
        await rag2.close()

    narrow_conn = duckdb.connect(db)
    typ = [
        row[1]
        for row in narrow_conn.execute("DESCRIBE unit_embeddings").fetchall()
        if row[0] == "vec"
    ][0]
    narrow_conn.close()
    assert "8" in str(typ)


@pytest.mark.asyncio
async def test_api_forwards_lang():
    report = ApiReport(library="lib", language="python", report="ok")
    storage = MagicMock()
    storage.close = AsyncMock()
    embedder = MagicMock()
    embedder.close = AsyncMock()
    with patch(
        "code_rag.api.client.create_stack",
        new=AsyncMock(return_value=(embedder, MagicMock(), MagicMock())),
    ), patch(
        "code_rag.api.client.open_db_connection",
        new=AsyncMock(return_value=storage),
    ), patch("code_rag.api.client.build_manager", return_value=MagicMock()), patch(
        "code_rag.api.client.run_api", new=AsyncMock(return_value=report)
    ) as mock_api:
        rag = CodeRAG()
        out = await rag.api("lib", lang=None)
        await rag.close()
    assert out is report
    mock_api.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_without_stack_is_noop():
    rag = CodeRAG()
    await rag.close()
    assert rag._embedder is None
    assert rag._storage is None


@pytest.mark.asyncio
async def test_close_closes_stray_storage():
    storage = AsyncMock()
    rag = CodeRAG(db=None, root=None)
    rag._storage = storage
    await rag.close()
    storage.close.assert_awaited_once()
    assert rag._storage is None
