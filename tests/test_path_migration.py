import pytest

from code_rag.core.models import KnowledgeUnit, Relation, RelationType, UnitKind
from code_rag.storage.db_connection import AccessMode, open_db_connection
from code_rag.storage.duckdb_impl import PATHS_MIGRATED_KEY
from tests.embedder_stubs import StubEmbedder


def _unit(unit_id: str, path: str, code_hash: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        name=unit_id.rsplit(":", 1)[-1],
        kind=UnitKind.FUNCTION,
        path=path,
        code_hash=code_hash,
        summary="kept",
    )


@pytest.mark.asyncio
async def test_commit_path_migration_rewrites_ids_and_sets_mark(tmp_path):
    old = "C:/old/proj/src/a.py"
    storage = await open_db_connection(
        tmp_path / "t.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        unit = _unit(f"{old}:alpha", old, "h1")
        unit.relations = [
            Relation(
                from_id=f"{old}:alpha", to_id=f"{old}:beta", type=RelationType.CALLS
            )
        ]
        await storage.upsert_unit(unit, vector=[0.0] * 384)
        other = _unit(f"{old}:beta", old, "h2")
        await storage.upsert_unit(other, vector=[0.0] * 384)

        assert await storage.paths_migration_done() is False
        await storage.commit_path_migration({old: "src/a.py"})

        rows = storage.conn.execute("SELECT id, path FROM units ORDER BY id").fetchall()
        assert rows == [("src/a.py:alpha", "src/a.py"), ("src/a.py:beta", "src/a.py")]
        embeds = storage.conn.execute(
            "SELECT id FROM unit_embeddings ORDER BY id"
        ).fetchall()
        assert [row[0] for row in embeds] == ["src/a.py:alpha", "src/a.py:beta"]
        rels = storage.conn.execute("SELECT from_id, to_id FROM relations").fetchall()
        assert rels == [("src/a.py:alpha", "src/a.py:beta")]
        assert await storage.paths_migration_done() is True
        mark = storage.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            [PATHS_MIGRATED_KEY],
        ).fetchone()
        assert mark[0] == "1"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_failed_rewrite_does_not_set_mark(tmp_path):
    storage = await open_db_connection(
        tmp_path / "t.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        old_a = "C:/old/a.py"
        old_b = "C:/old/b.py"
        await storage.upsert_unit(_unit(f"{old_a}:alpha", old_a, "h1"))
        await storage.upsert_unit(_unit(f"{old_b}:alpha", old_b, "h2"))
        with pytest.raises(Exception):
            await storage.commit_path_migration({old_a: "src/a.py", old_b: "src/a.py"})
        paths = storage.conn.execute("SELECT path FROM units ORDER BY path").fetchall()
        assert [row[0] for row in paths] == [old_a, old_b]
        assert await storage.paths_migration_done() is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_empty_mapping_only_writes_the_mark(tmp_path):
    storage = await open_db_connection(
        tmp_path / "t.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        await storage.commit_path_migration({})
        assert await storage.paths_migration_done() is True
    finally:
        await storage.close()
