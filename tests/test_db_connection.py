import threading
import pytest
from code_rag.core.error_codes import ErrorCode
from code_rag.core.exceptions import StorageBusyError, StorageError
from code_rag.storage import duckdb_impl
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
    import duckdb

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        # Message must not matter — only IOException type triggers busy retry.
        raise duckdb.IOException("platform-specific sharing violation text")

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
    import duckdb

    real_connect = duckdb.connect
    state = {"n": 0}

    def flaky(path, **kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise duckdb.IOException("temporary lock")
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
async def test_non_io_connect_error_is_storage_error_without_retry(
    monkeypatch, tmp_path
):
    import duckdb

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise duckdb.ConnectionException("different configuration")

    monkeypatch.setattr("duckdb.connect", boom)
    with pytest.raises(StorageError, match="Failed to open storage"):
        await open_db_connection(
            tmp_path / "x.db",
            StubEmbedder(dim=384),
            mode=AccessMode.READ_WRITE,
            connect_timeout_seconds=1.0,
        )
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_plain_exception_is_not_treated_as_busy(monkeypatch, tmp_path):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("Could not set lock on file")  # text alone is not enough

    monkeypatch.setattr("duckdb.connect", boom)
    with pytest.raises(StorageError, match="Failed to open storage"):
        await open_db_connection(
            tmp_path / "x.db",
            StubEmbedder(dim=384),
            mode=AccessMode.READ_WRITE,
            connect_timeout_seconds=1.0,
        )
    assert calls["n"] == 1


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
async def test_thread_affinity_bind_embeddings_and_finalize(tmp_path, monkeypatch):
    """Open-finalize (_refresh_dirty_from_meta) and ensure_embeddings_bound
    (_bind_embeddings/bind_embedder_dimension) SQL must run on the
    connection's dedicated executor thread, never the event-loop thread.
    """
    ids = set()
    main_thread_id = threading.get_ident()
    original_meta_get = duckdb_impl._meta_get  # pylint: disable=protected-access

    def spy_meta_get(conn, key):
        ids.add(threading.get_ident())
        return original_meta_get(conn, key)

    monkeypatch.setattr(duckdb_impl, "_meta_get", spy_meta_get)

    emb = StubEmbedder(dim=384)
    storage = await open_db_connection(
        tmp_path / "bind_aff.db",
        emb,
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        await storage.ensure_embeddings_bound()
    finally:
        await storage.close()

    assert ids, "expected _meta_get to be exercised by finalize/bind"
    assert main_thread_id not in ids
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


@pytest.mark.asyncio
async def test_live_cross_process_lock_raises_storage_busy(tmp_path):
    """Real second-process RW holder must surface as StorageBusyError.

    Uses subprocess (not multiprocessing) so Windows spawn works from pytest.
    """
    import subprocess
    import sys
    import time

    db_path = tmp_path / "live_busy.db"
    ready = tmp_path / "holder.ready"
    stop = tmp_path / "holder.stop"
    holder_script = tmp_path / "holder.py"
    holder_script.write_text(
        "\n".join(
            [
                "import time",
                "from pathlib import Path",
                "import duckdb",
                f"db = Path(r'{db_path}')",
                f"ready = Path(r'{ready}')",
                f"stop = Path(r'{stop}')",
                "conn = duckdb.connect(str(db))",
                "conn.execute('CREATE TABLE IF NOT EXISTS t(i INTEGER)')",
                "ready.write_text('1', encoding='utf-8')",
                "for _ in range(200):",
                "    if stop.exists():",
                "        break",
                "    time.sleep(0.05)",
                "conn.close()",
                "",
            ]
        ),
        encoding="utf-8",
    )

    child = subprocess.Popen(  # nosec B603
        [sys.executable, str(holder_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    emb = StubEmbedder(dim=384)
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.05)
        else:
            pytest.fail("holder process did not become ready")

        with pytest.raises(StorageBusyError) as ei:
            await open_db_connection(
                db_path,
                emb,
                mode=AccessMode.READ_WRITE,
                connect_timeout_seconds=0.05,
            )
        assert ei.value.code is ErrorCode.STORAGE_BUSY

        with pytest.raises(StorageBusyError):
            await open_db_connection(
                db_path,
                None,
                mode=AccessMode.READ_ONLY,
                connect_timeout_seconds=0.05,
            )
    finally:
        stop.write_text("1", encoding="utf-8")
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        await emb.close()
