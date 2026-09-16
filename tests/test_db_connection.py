import threading
import pytest
from code_rag.core.error_codes import ErrorCode
from code_rag.core.exceptions import StorageBusyError, StorageError
from code_rag.storage.db_connection import AccessMode, open_db_connection
from tests.embedder_stubs import StubEmbedder


@pytest.mark.asyncio
async def test_ro_missing_file_does_not_create(tmp_path):
    path = tmp_path / "missing.db"
    emb = StubEmbedder(dim=384)
    with pytest.raises(StorageError):
        await open_db_connection(
            path, emb, mode=AccessMode.READ_ONLY, connect_timeout_seconds=0
        )
    assert not path.exists()
    assert emb.closed is False


@pytest.mark.asyncio
async def test_rw_creates_file_and_close_does_not_close_embedder(tmp_path):
    path = tmp_path / ".coderag.db"
    emb = StubEmbedder(dim=384)
    storage = await open_db_connection(
        path, emb, mode=AccessMode.READ_WRITE, connect_timeout_seconds=0
    )
    try:
        assert path.exists()
    finally:
        await storage.close()
    assert emb.closed is False
    await emb.close()
    assert emb.closed is True


@pytest.mark.asyncio
async def test_busy_timeout_raises_storage_busy(monkeypatch, tmp_path):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise Exception("IO Error: Could not set lock on file")

    monkeypatch.setattr("duckdb.connect", boom)
    with pytest.raises(StorageBusyError) as ei:
        await open_db_connection(
            tmp_path / "x.db",
            StubEmbedder(dim=384),
            mode=AccessMode.READ_WRITE,
            connect_timeout_seconds=0.05,
        )
    assert ei.value.code is ErrorCode.STORAGE_BUSY
    assert calls["n"] >= 1


@pytest.mark.asyncio
async def test_busy_retry_then_success(monkeypatch, tmp_path):
    real_connect = __import__("duckdb").connect
    state = {"n": 0}

    def flaky(path, **kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise Exception("IO Error: Could not set lock on file")
        return real_connect(path, **kwargs)

    monkeypatch.setattr("duckdb.connect", flaky)
    emb = StubEmbedder(dim=384)
    storage = await open_db_connection(
        tmp_path / "ok.db",
        emb,
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=2.0,
    )
    try:
        assert state["n"] >= 3
    finally:
        await storage.close()
    await emb.close()


@pytest.mark.asyncio
async def test_thread_affinity_connect_execute_close(tmp_path):
    emb = StubEmbedder(dim=384)
    storage = await open_db_connection(
        tmp_path / "aff.db",
        emb,
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    ids = set()

    def grab():
        ids.add(threading.get_ident())
        return storage.conn.execute("SELECT 1").fetchone()

    try:
        await storage._with_conn(grab)
        await storage._with_conn(grab)
    finally:
        # close must run on same executor thread — record via patched close if needed
        await storage.close()
    assert len(ids) == 1
    await emb.close()


@pytest.mark.asyncio
async def test_ro_rejects_wipe(tmp_path):
    path = tmp_path / "ro.db"
    emb = StubEmbedder(dim=384)
    rw = await open_db_connection(
        path, emb, mode=AccessMode.READ_WRITE, connect_timeout_seconds=0
    )
    await rw.close()
    with pytest.raises(StorageError, match="wipe=True is not supported"):
        await open_db_connection(
            path,
            None,
            mode=AccessMode.READ_ONLY,
            connect_timeout_seconds=0,
            wipe=True,
        )
    await emb.close()


@pytest.mark.asyncio
async def test_metadata_only_ro_open_without_embedder(tmp_path):
    # Create RW first with embedder so file exists
    emb = StubEmbedder(dim=384)
    path = tmp_path / "meta.db"
    rw = await open_db_connection(
        path, emb, mode=AccessMode.READ_WRITE, connect_timeout_seconds=0
    )
    await rw.close()
    ro = await open_db_connection(
        path, None, mode=AccessMode.READ_ONLY, connect_timeout_seconds=0
    )
    try:
        assert await ro.get_dependency_path("missing-lib") is None
    finally:
        await ro.close()
    await emb.close()
