from unittest.mock import AsyncMock, MagicMock

import pytest

from code_rag.api.client import CodeRAG
from code_rag.core.exceptions import StorageBusyError
from code_rag.storage.db_connection import AccessMode


@pytest.mark.asyncio
async def test_api_python_does_not_open_db_or_create_embedder(monkeypatch):
    opened = {"n": 0}
    embeds = {"n": 0}

    async def fake_open(*a, **k):
        opened["n"] += 1
        raise AssertionError("should not open")

    async def fake_create_embedder(*a, **k):
        embeds["n"] += 1
        raise AssertionError("should not embed")

    monkeypatch.setattr("code_rag.api.client.open_db_connection", fake_open)
    monkeypatch.setattr(
        "code_rag.intelligence.factory.create_embedder", fake_create_embedder
    )
    monkeypatch.setattr(
        "code_rag.services.factory.create_embedder", fake_create_embedder
    )
    monkeypatch.setattr(
        "code_rag.discovery.manager.DiscoveryManager.extract_api",
        AsyncMock(return_value="ok"),
    )

    rag = CodeRAG(db="unused.db")
    try:
        out = await rag.api("pydantic")
    finally:
        await rag.close()

    assert out.report == "ok"
    assert opened["n"] == 0
    assert embeds["n"] == 0


@pytest.mark.asyncio
async def test_api_java_opens_ro_without_embedder_and_closes(monkeypatch):
    opened: list[dict] = []
    storage = MagicMock()
    storage.close = AsyncMock()
    storage.get_dependency_path = AsyncMock(return_value=None)

    async def fake_open(
        path, embedder, *, mode, connect_timeout_seconds=5.0, wipe=False
    ):
        opened.append({"embedder": embedder, "mode": mode, "wipe": wipe})
        return storage

    async def fake_create_embedder(*a, **k):
        raise AssertionError("should not embed")

    monkeypatch.setattr("code_rag.api.client.open_db_connection", fake_open)
    monkeypatch.setattr(
        "code_rag.services.factory.create_embedder", fake_create_embedder
    )

    rag = CodeRAG(db="unused.db")
    try:
        out = await rag.api("guava", lang="java")
    finally:
        await rag.close()

    assert len(opened) == 1
    assert opened[0]["embedder"] is None
    assert opened[0]["mode"] is AccessMode.READ_ONLY
    assert "Could not find cached JAR" in out.report
    storage.close.assert_awaited()
    assert rag._storage is None


@pytest.mark.asyncio
async def test_api_java_cache_miss_message(tmp_path):
    rag = CodeRAG(db=str(tmp_path / "missing.db"), root=tmp_path)
    try:
        out = await rag.api("guava", lang="java")
    finally:
        await rag.close()
    assert "Could not find cached JAR" in out.report


@pytest.mark.asyncio
async def test_api_java_locked_file_raises_storage_busy(monkeypatch, tmp_path):
    async def fake_open(*a, **k):
        raise StorageBusyError()

    monkeypatch.setattr("code_rag.api.client.open_db_connection", fake_open)
    rag = CodeRAG(db=str(tmp_path / "busy.db"), root=tmp_path)
    with pytest.raises(StorageBusyError):
        await rag.api("guava", lang="java")
    await rag.close()
