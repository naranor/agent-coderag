import argparse
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from code_rag.api.client import CodeRAG
from code_rag.api.models import ApiReport, SyncResult
from code_rag.entry import cli
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.storage.db_connection import AccessMode, open_db_connection
from tests.embedder_stubs import StubEmbedder


async def _seed_index(path: Path, embedder: StubEmbedder) -> None:
    """Create a real index file so read-only opens can bind embeddings."""
    storage = await open_db_connection(
        path, embedder, mode=AccessMode.READ_WRITE, connect_timeout_seconds=0
    )
    try:
        await storage.ensure_embeddings_bound()
    finally:
        await storage.close()


@pytest.fixture(name="stub_stack")
def stub_stack_fixture(monkeypatch):
    """Patch the process-scoped stack factory to a counted stub embedder."""
    embedder = StubEmbedder(dim=8)
    calls = {"create_embedder": 0}

    async def fake_create_embedder(config, onnx_path=None):  # noqa: ARG001
        calls["create_embedder"] += 1
        return embedder

    monkeypatch.setattr(
        "code_rag.services.factory.create_embedder", fake_create_embedder
    )
    monkeypatch.setattr(
        "code_rag.services.factory.DistillerConfig.load", lambda: DistillerConfig()
    )
    return embedder, calls


@pytest.mark.asyncio
async def test_two_searches_reuse_embedder_and_reopen_db(
    tmp_path, monkeypatch, stub_stack
):
    embedder, calls = stub_stack
    db = tmp_path / "life.db"
    await _seed_index(db, embedder)

    opens: list[dict] = []
    closes = {"n": 0}
    real_open = open_db_connection

    async def counting_open(
        path, emb, *, mode, connect_timeout_seconds=5.0, wipe=False
    ):
        opens.append({"mode": mode, "wipe": wipe})
        storage = await real_open(
            path,
            emb,
            mode=mode,
            connect_timeout_seconds=connect_timeout_seconds,
            wipe=wipe,
        )
        inner_close = storage.close

        async def counting_close():
            closes["n"] += 1
            await inner_close()

        storage.close = counting_close
        return storage

    monkeypatch.setattr("code_rag.api.client.open_db_connection", counting_open)

    rag = CodeRAG(db=str(db), root=tmp_path)
    await rag.search("first", limit=1)
    await rag.search("second", limit=1)

    assert calls["create_embedder"] == 1
    assert len(opens) == 2
    assert all(entry["mode"] is AccessMode.READ_ONLY for entry in opens)
    assert closes["n"] == 2
    assert embedder.closed is False

    await rag.close()
    assert embedder.closed is True


@pytest.mark.asyncio
async def test_overlapping_ops_serialize(tmp_path, monkeypatch, stub_stack):
    embedder, calls = stub_stack
    db = tmp_path / "serial.db"
    await _seed_index(db, embedder)

    events: list[str] = []

    async def slow_search(manager, query, *, limit=5):  # noqa: ARG001
        events.append(f"start:{query}")
        await asyncio.sleep(0.02)
        events.append(f"end:{query}")
        return []

    monkeypatch.setattr("code_rag.api.client.run_search", slow_search)

    rag = CodeRAG(db=str(db), root=tmp_path)
    try:
        await asyncio.gather(rag.search("a"), rag.search("b"))
    finally:
        await rag.close()

    assert calls["create_embedder"] == 1
    assert len(events) == 4
    for index in (0, 2):
        query = events[index].split(":", 1)[1]
        assert events[index] == f"start:{query}"
        assert events[index + 1] == f"end:{query}"


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_op(tmp_path, monkeypatch, stub_stack):
    embedder, _ = stub_stack
    db = tmp_path / "inflight.db"
    await _seed_index(db, embedder)

    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_search(manager, query, *, limit=5):  # noqa: ARG001
        started.set()
        await release.wait()
        return []

    monkeypatch.setattr("code_rag.api.client.run_search", blocking_search)

    rag = CodeRAG(db=str(db), root=tmp_path)
    search_task = asyncio.create_task(rag.search("q"))
    await started.wait()
    close_task = asyncio.create_task(rag.close())
    await asyncio.sleep(0.05)

    assert close_task.done() is False
    assert embedder.closed is False

    release.set()
    await search_task
    await close_task
    assert embedder.closed is True


@pytest.mark.asyncio
async def test_rebuild_wipe_same_rw_connection(tmp_path, monkeypatch, stub_stack):
    embedder, _ = stub_stack
    db = tmp_path / "rebuild.db"
    await _seed_index(db, embedder)

    opens: list[dict] = []
    real_open = open_db_connection

    async def tracking_open(
        path, emb, *, mode, connect_timeout_seconds=5.0, wipe=False
    ):
        opens.append({"mode": mode, "wipe": wipe})
        return await real_open(
            path,
            emb,
            mode=mode,
            connect_timeout_seconds=connect_timeout_seconds,
            wipe=wipe,
        )

    monkeypatch.setattr("code_rag.api.client.open_db_connection", tracking_open)
    monkeypatch.setattr(
        "code_rag.api.client.run_rebuild",
        AsyncMock(return_value=SyncResult(status="success", indexed_files=0)),
    )

    rag = CodeRAG(db=str(db), root=tmp_path)
    try:
        await rag.rebuild()
    finally:
        await rag.close()

    assert len(opens) == 1
    assert opens[0]["wipe"] is True
    assert opens[0]["mode"] is AccessMode.READ_WRITE


@pytest.mark.asyncio
async def test_logs_resolved_db_path(tmp_path, caplog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="code_rag.api.client")

    CodeRAG(db=None, root=tmp_path)

    matches = [
        record
        for record in caplog.records
        if ".coderag.db" in record.getMessage() or "code_rag.db" in record.getMessage()
    ]
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_cli_commands_use_facade_not_legacy_factory(tmp_path, monkeypatch):
    assert not hasattr(cli, "get_manager")
    assert not hasattr(cli, "create_manager")

    constructed: list[dict] = []

    class FakeRAG:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.sync = AsyncMock(
                return_value=SyncResult(status="success", indexed_files=1)
            )
            self.search = AsyncMock(return_value=[])
            self.api = AsyncMock(
                return_value=ApiReport(library="lib", language="python", report="r")
            )
            self.rebuild = AsyncMock(
                return_value=SyncResult(status="success", indexed_files=1)
            )
            self.close = AsyncMock()

    monkeypatch.setattr(cli, "CodeRAG", FakeRAG)
    db = str(tmp_path / "cli.db")

    await cli.sync_cmd(
        argparse.Namespace(
            db=db,
            onnx=None,
            json=True,
            path=None,
            all=True,
            force=False,
            connect_timeout=7.0,
        )
    )
    await cli.search_cmd(
        argparse.Namespace(
            db=db, onnx=None, json=True, query="q", limit=2, connect_timeout=7.0
        )
    )
    await cli.api_cmd(
        argparse.Namespace(
            db=db, onnx=None, json=True, library="lib", lang=None, connect_timeout=7.0
        )
    )

    assert len(constructed) == 3
    assert all(kwargs["db"] == db for kwargs in constructed)
    assert all(kwargs["connect_timeout_seconds"] == 7.0 for kwargs in constructed)
