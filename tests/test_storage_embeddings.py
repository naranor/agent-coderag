from unittest.mock import patch

import duckdb
import pytest

from code_rag.core.constants import EMBEDDING_PROBE_TEXT, LOCAL_EMBEDDING_MODEL_ID
from code_rag.core.error_codes import ErrorCode
from code_rag.core.exceptions import IntelligenceError, StorageError
from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.storage.duckdb_impl import (
    DuckDBStorage,
    _meta_get,
    _parse_positive_dim,
    _schema_vec_width,
    bind_embedder_dimension,
    float_vec_sql_type,
)
from tests.embedder_stubs import StubEmbedder, UnboundStubEmbedder


def _unit(uid="u1", summary="s", code_hash="h"):
    return KnowledgeUnit(
        id=uid,
        name=uid,
        kind=UnitKind.FUNCTION,
        path="p.py",
        code_hash=code_hash,
        summary=summary,
    )


def test_float_vec_sql_type_int_only():
    assert float_vec_sql_type(8) == "FLOAT[8]"
    with pytest.raises(StorageError, match="positive int"):
        float_vec_sql_type(True)  # type: ignore[arg-type]
    with pytest.raises(StorageError, match="positive int"):
        float_vec_sql_type("384")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_open_requires_embedder(tmp_path):
    with pytest.raises(StorageError, match="embedder is required"):
        await DuckDBStorage.open(str(tmp_path / "t.db"), None)


@pytest.mark.asyncio
async def test_open_does_not_probe_or_create_embeddings(tmp_path):
    stub = UnboundStubEmbedder(probe_dim=16, model_id="remote-16")
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    assert stub.aembed_calls == []
    row = storage.conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE lower(table_name)='unit_embeddings'"
    ).fetchone()
    assert row is None
    await storage.close()
    assert stub.closed is False
    await stub.close()
    assert stub.closed is True


@pytest.mark.asyncio
async def test_ensure_bound_probes_and_creates_table(tmp_path):
    stub = UnboundStubEmbedder(probe_dim=16, model_id="remote-16")
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    await storage.ensure_embeddings_bound()
    assert stub.dimension == 16
    assert stub.aembed_calls[0] == [EMBEDDING_PROBE_TEXT]
    row = storage.conn.execute(
        "SELECT value FROM index_meta WHERE key='embedding_dim'"
    ).fetchone()
    assert row[0] == "16"
    desc = storage.conn.execute("DESCRIBE unit_embeddings").fetchall()
    types = {r[0]: r[1] for r in desc}
    assert "16" in str(types["vec"])
    await storage.close()


@pytest.mark.asyncio
async def test_upsert_does_not_call_embedder(tmp_path):
    stub = StubEmbedder(dim=8)
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    stub.aembed_calls.clear()
    await storage.upsert_unit(_unit(), vector=[0.1] * 8)
    assert stub.aembed_calls == []
    assert await storage.has_embedding("u1") is True
    await storage.close()


@pytest.mark.asyncio
async def test_search_uses_embedder_dimension_not_384(tmp_path):
    stub = StubEmbedder(dim=8, model_id="stub")
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    await storage.upsert_unit(_unit(), vector=[1.0] + [0.0] * 7)
    stub.aembed_calls.clear()
    hits = await storage.search_units("q", limit=1)
    assert hits[0].id == "u1"
    assert stub.aembed_calls[-1] == ["q"]
    await storage.close()


@pytest.mark.asyncio
async def test_no_ilike_fallback_without_vector_row(tmp_path):
    stub = StubEmbedder(dim=8)
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    await storage.upsert_unit(_unit())  # no vector
    hits = await storage.search_units("u1")
    assert hits == []
    await storage.close()


@pytest.mark.asyncio
async def test_failed_bind_on_dim_mismatch(tmp_path):
    path = str(tmp_path / "t.db")
    stub16 = StubEmbedder(dim=16, model_id="m16")
    storage = await DuckDBStorage.open(path, stub16)
    await storage.ensure_embeddings_bound()
    await storage.close()
    stub8 = StubEmbedder(dim=8, model_id="m8")
    storage = await DuckDBStorage.open(path, stub8, wipe=False)
    with pytest.raises(StorageError, match="Run rebuild") as ei:
        await storage.ensure_embeddings_bound()
    assert ei.value.code is ErrorCode.EMBEDDING_MISMATCH
    await storage.close()
    storage = await DuckDBStorage.open(path, stub16, wipe=False)
    await storage.ensure_embeddings_bound()
    await storage.close()
    storage = await DuckDBStorage.open(path, stub8, wipe=True)
    await storage.ensure_embeddings_bound()
    await storage.close()


@pytest.mark.asyncio
async def test_dim_mismatch_search_sync_storage_error(tmp_path):
    path = str(tmp_path / "t.db")
    stub16 = StubEmbedder(dim=16, model_id="m16")
    storage = await DuckDBStorage.open(path, stub16)
    await storage.ensure_embeddings_bound()
    await storage.close()
    stub8 = StubEmbedder(dim=8, model_id="m8")
    storage = await DuckDBStorage.open(path, stub8, wipe=False)
    with pytest.raises(StorageError, match="Run rebuild") as ei:
        await storage.ensure_embeddings_bound()
    assert ei.value.code is ErrorCode.EMBEDDING_MISMATCH
    await storage.close()


@pytest.mark.asyncio
async def test_rebuild_wipe_recreates_float_n(tmp_path):
    path = str(tmp_path / "t.db")
    stub16 = StubEmbedder(dim=16, model_id="m16")
    storage = await DuckDBStorage.open(path, stub16)
    await storage.upsert_unit(_unit(), vector=[0.0] * 16)
    await storage.close()
    stub8 = StubEmbedder(dim=8, model_id="m8")
    storage = await DuckDBStorage.open(path, stub8, wipe=True)
    await storage.ensure_embeddings_bound()
    desc = storage.conn.execute("DESCRIBE unit_embeddings").fetchall()
    types = {r[0]: r[1] for r in desc}
    assert "8" in str(types["vec"])
    units_left = storage.conn.execute("SELECT count(*) FROM units").fetchone()[0]
    assert units_left == 1
    await storage.close()


@pytest.mark.asyncio
async def test_legacy_without_meta_infers_384(tmp_path):
    path = str(tmp_path / "legacy.db")
    conn = duckdb.connect(path)
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    conn.execute(
        "CREATE TABLE units (id VARCHAR PRIMARY KEY, kind VARCHAR, name VARCHAR, path VARCHAR, signature VARCHAR, docstring VARCHAR, summary VARCHAR, code_hash VARCHAR, tags VARCHAR[], metadata JSON)"
    )
    conn.execute(
        "CREATE TABLE unit_embeddings (id VARCHAR PRIMARY KEY, vec FLOAT[384])"
    )
    conn.close()
    stub = StubEmbedder(dim=384, model_id=LOCAL_EMBEDDING_MODEL_ID)
    storage = await DuckDBStorage.open(path, stub, wipe=False)
    await storage.ensure_embeddings_bound()
    dim = storage.conn.execute(
        "SELECT value FROM index_meta WHERE key='embedding_dim'"
    ).fetchone()[0]
    model = storage.conn.execute(
        "SELECT value FROM index_meta WHERE key='embedding_model'"
    ).fetchone()[0]
    assert dim == "384"
    assert model == LOCAL_EMBEDDING_MODEL_ID
    await storage.close()


@pytest.mark.asyncio
async def test_legacy_without_meta_dim_mismatch_raises_storage_error(tmp_path):
    path = str(tmp_path / "legacy_mismatch.db")
    conn = duckdb.connect(path)
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    conn.execute(
        "CREATE TABLE units (id VARCHAR PRIMARY KEY, kind VARCHAR, name VARCHAR, path VARCHAR, signature VARCHAR, docstring VARCHAR, summary VARCHAR, code_hash VARCHAR, tags VARCHAR[], metadata JSON)"
    )
    conn.execute(
        "CREATE TABLE unit_embeddings (id VARCHAR PRIMARY KEY, vec FLOAT[384])"
    )
    conn.close()
    stub = StubEmbedder(dim=8, model_id=LOCAL_EMBEDDING_MODEL_ID)
    storage = await DuckDBStorage.open(path, stub, wipe=False)
    with pytest.raises(
        StorageError,
        match=r"Embedding dimension mismatch \(index=384, embedder=8\)\. Run rebuild\.",
    ):
        await storage.ensure_embeddings_bound()
    await storage.close()


@pytest.mark.asyncio
async def test_corrupt_meta_storage_error(tmp_path):
    stub = StubEmbedder(dim=384)
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    await storage.ensure_embeddings_bound()
    storage.conn.execute(
        "INSERT OR REPLACE INTO index_meta VALUES ('embedding_dim', 'nope')"
    )
    await storage.close()
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), StubEmbedder(dim=384))
    with pytest.raises(StorageError, match="Corrupt embedding metadata") as ei:
        await storage.ensure_embeddings_bound()
    assert ei.value.code is ErrorCode.STORAGE_CORRUPT
    await storage.close()


@pytest.mark.asyncio
async def test_dirty_model_same_dim_search_warns(tmp_path, capsys):
    path = str(tmp_path / "t.db")
    a = StubEmbedder(dim=8, model_id="model-a")
    storage = await DuckDBStorage.open(path, a)
    await storage.upsert_unit(_unit(), vector=[0.0] * 8)
    await storage.close()
    b = StubEmbedder(dim=8, model_id="model-b")
    storage = await DuckDBStorage.open(path, b, wipe=False)
    assert storage.embedding_model_dirty is True
    hits = await storage.search_units("q")
    assert hits[0].id == "u1"
    err = capsys.readouterr().err
    assert "embedding model id changed" in err
    await storage.close()


@pytest.mark.asyncio
async def test_custom_onnx_dim_mismatch_is_storage_error(tmp_path):
    path = str(tmp_path / "t.db")
    local384 = StubEmbedder(dim=384, model_id=LOCAL_EMBEDDING_MODEL_ID)
    storage = await DuckDBStorage.open(path, local384)
    await storage.ensure_embeddings_bound()
    await storage.close()
    custom8 = StubEmbedder(dim=8, model_id=LOCAL_EMBEDDING_MODEL_ID)
    storage = await DuckDBStorage.open(path, custom8, wipe=False)
    with pytest.raises(StorageError, match="Run rebuild"):
        await storage.ensure_embeddings_bound()
    await storage.close()


@pytest.mark.asyncio
async def test_model_match_skips_probe(tmp_path):
    path = str(tmp_path / "t.db")
    stub = UnboundStubEmbedder(probe_dim=8, model_id="remote-8")
    storage = await DuckDBStorage.open(path, stub)
    await storage.ensure_embeddings_bound()
    await storage.close()
    stub2 = UnboundStubEmbedder(probe_dim=8, model_id="remote-8")
    storage = await DuckDBStorage.open(path, stub2)
    await storage.ensure_embeddings_bound()
    probe_calls = [c for c in stub2.aembed_calls if c == [EMBEDDING_PROBE_TEXT]]
    assert probe_calls == []
    await storage.close()


@pytest.mark.asyncio
async def test_non_vector_ops_do_not_bind(tmp_path):
    stub = UnboundStubEmbedder(probe_dim=16, model_id="remote-16")
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    await storage.set_dependency_path("lib", "/tmp/lib.jar")
    assert await storage.get_dependency_path("lib") == "/tmp/lib.jar"
    await storage.upsert_unit(_unit())
    got = await storage.get_unit("u1")
    assert got is not None
    assert stub.aembed_calls == []
    await storage.close()


def test_parse_positive_dim_rejects_zero():
    with pytest.raises(StorageError, match="Corrupt embedding metadata") as ei:
        _parse_positive_dim("0")
    assert ei.value.code is ErrorCode.STORAGE_CORRUPT


def test_schema_vec_width_without_float_array():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE unit_embeddings (id VARCHAR PRIMARY KEY, vec VARCHAR)")
    assert _schema_vec_width(conn) is None
    conn.close()


def test_meta_get_without_index_meta_table():
    conn = duckdb.connect(":memory:")
    assert _meta_get(conn, "embedding_dim") is None
    conn.close()


@pytest.mark.asyncio
async def test_probe_empty_rows_raises_count_mismatch():
    class EmptyProbe(UnboundStubEmbedder):
        async def aembed(self, texts):
            self.aembed_calls.append(list(texts))
            return []

    stub = EmptyProbe(probe_dim=8, model_id="remote")
    conn = duckdb.connect(":memory:")
    with pytest.raises(IntelligenceError, match="count mismatch"):
        await bind_embedder_dimension(conn, stub, wiped=True)
    conn.close()


@pytest.mark.asyncio
async def test_corrupt_schema_vs_meta_raises(tmp_path):
    path = str(tmp_path / "t.db")
    stub = StubEmbedder(dim=8, model_id="m8")
    storage = await DuckDBStorage.open(path, stub)
    await storage.ensure_embeddings_bound()
    storage.conn.execute(
        "INSERT OR REPLACE INTO index_meta VALUES ('embedding_dim', '384')"
    )
    await storage.close()
    storage = await DuckDBStorage.open(path, StubEmbedder(dim=8, model_id="m8"))
    with pytest.raises(StorageError, match="Corrupt embedding metadata") as ei:
        await storage.ensure_embeddings_bound()
    assert ei.value.code is ErrorCode.STORAGE_CORRUPT
    await storage.close()


@pytest.mark.asyncio
async def test_legacy_incomplete_meta_probes_remote(tmp_path):
    path = str(tmp_path / "legacy_remote.db")
    conn = duckdb.connect(path)
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    conn.execute(
        "CREATE TABLE units (id VARCHAR PRIMARY KEY, kind VARCHAR, name VARCHAR, path VARCHAR, signature VARCHAR, docstring VARCHAR, summary VARCHAR, code_hash VARCHAR, tags VARCHAR[], metadata JSON)"
    )
    conn.execute(
        "CREATE TABLE unit_embeddings (id VARCHAR PRIMARY KEY, vec FLOAT[384])"
    )
    conn.close()
    stub = UnboundStubEmbedder(probe_dim=384, model_id="remote-384")
    storage = await DuckDBStorage.open(path, stub, wipe=False)
    await storage.ensure_embeddings_bound()
    assert stub.aembed_calls[0] == [EMBEDDING_PROBE_TEXT]
    assert stub.dimension == 384
    await storage.close()


@pytest.mark.asyncio
async def test_ensure_bound_double_checked_lock(tmp_path):
    import asyncio as aio

    stub = UnboundStubEmbedder(probe_dim=8, model_id="remote-8")
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    original = storage._bind_embeddings

    async def slow_bind():
        await aio.sleep(0.05)
        await original()

    storage._bind_embeddings = slow_bind
    await aio.gather(
        storage.ensure_embeddings_bound(),
        storage.ensure_embeddings_bound(),
    )
    assert storage._embeddings_bound is True
    await storage.close()


@pytest.mark.asyncio
async def test_open_failure_closes_connection(tmp_path):
    with patch(
        "code_rag.storage.duckdb_impl._ensure_base_tables",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await DuckDBStorage.open(str(tmp_path / "t.db"), StubEmbedder())


@pytest.mark.asyncio
async def test_open_failure_ignores_close_error():
    class BoomConn:
        def close(self):
            raise RuntimeError("close fail")

    with patch(
        "code_rag.storage.duckdb_impl.duckdb.connect", return_value=BoomConn()
    ), patch(
        "code_rag.storage.duckdb_impl._ensure_base_tables",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await DuckDBStorage.open("ignored.db", StubEmbedder())


@pytest.mark.asyncio
async def test_list_units_and_mark_synced(tmp_path):
    stub = StubEmbedder(dim=8, model_id="m8")
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)
    await storage.upsert_unit(_unit(), vector=[0.0] * 8)
    listed = await storage.list_units()
    assert listed[0].id == "u1"
    storage._embedding_model_dirty = True
    await storage.mark_embedding_model_synced()
    assert storage.embedding_model_dirty is False
    model = storage.conn.execute(
        "SELECT value FROM index_meta WHERE key='embedding_model'"
    ).fetchone()[0]
    assert model == "m8"
    await storage.close()


@pytest.mark.asyncio
async def test_close_skips_missing_embedder_and_conn(tmp_path):
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), StubEmbedder())
    conn = storage.conn
    storage._embedder = None
    storage.conn = None
    await storage.close()
    conn.close()


@pytest.mark.asyncio
async def test_concurrent_upsert_and_get_keep_kind(tmp_path):
    import asyncio as aio

    stub = StubEmbedder(dim=8)
    storage = await DuckDBStorage.open(str(tmp_path / "t.db"), stub)

    async def one(i: int) -> None:
        uid = f"u{i}"
        await storage.upsert_unit(_unit(uid=uid, summary=f"s{i}"), vector=[0.0] * 8)
        got = await storage.get_unit(uid)
        assert got is not None
        assert got.kind == UnitKind.FUNCTION
        assert got.name == uid
        assert await storage.has_embedding(uid) is True

    await aio.gather(*[one(i) for i in range(24)])
    listed = await storage.list_units()
    assert len(listed) == 24
    await storage.close()
