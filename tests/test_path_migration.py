from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_rag.core.models import KnowledgeUnit, Relation, RelationType, UnitKind
from code_rag.parsers.multi_parser import MultiParser
from code_rag.services.indexing import IndexStack
from code_rag.services.path_migration import (
    claim_relative_paths,
    migrate_absolute_paths,
)
from code_rag.services.sync import SyncOptions, run_sync
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
async def test_failed_rewrite_keeps_earlier_path_and_skips_mark(tmp_path):
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
        await storage.commit_rewritten_path(old_a, "src/a.py")
        with pytest.raises(Exception):
            await storage.commit_rewritten_path(old_b, "src/a.py")
        paths = {
            row[0] for row in storage.conn.execute("SELECT path FROM units").fetchall()
        }
        assert paths == {"src/a.py", old_b}
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
        await storage.mark_paths_migrated()
        assert await storage.paths_migration_done() is True
    finally:
        await storage.close()


def test_under_root_claim_ignores_hashes(tmp_path: Path):
    root = tmp_path / "proj"
    target = root / "src" / "a.py"
    old = str(target.resolve())
    other = root / "other.py"
    mapping = claim_relative_paths(
        {old: Counter({"stale": 1})},
        root,
        [(other, Counter({"stale": 1}))],
    )
    assert mapping == {old: "src/a.py"}


def test_walk_order_claims_the_first_equal_multiset(tmp_path: Path):
    root = tmp_path / "proj"
    first = root / "first.py"
    second = root / "second.py"
    old = str((tmp_path / "elsewhere" / "old.py").resolve())
    mapping = claim_relative_paths(
        {old: Counter({"h": 2, "g": 1})},
        root,
        [
            (first, Counter({"h": 2, "g": 1})),
            (second, Counter({"h": 2, "g": 1})),
        ],
    )
    assert mapping == {old: "first.py"}


def test_taken_relative_path_is_not_claimed_again(tmp_path: Path):
    root = tmp_path / "proj"
    target = (root / "src" / "a.py").resolve()
    old_under = str(target)
    old_outside = str((tmp_path / "other" / "a.py").resolve())
    mapping = claim_relative_paths(
        {old_under: Counter({"h": 1}), old_outside: Counter({"h": 1})},
        root,
        [(target, Counter({"h": 1}))],
    )
    assert mapping == {old_under: "src/a.py"}


@pytest.mark.asyncio
async def test_sync_migrates_once_without_calling_llm(tmp_path: Path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    source_text = "def alpha():\n    return 1\n"
    old_file = old_root / "src" / "a.py"
    old_file.parent.mkdir(parents=True)
    old_file.write_text(source_text, encoding="utf-8")

    parsed = await MultiParser().distill_file(str(old_file.resolve()))
    assert parsed
    old_path = str(old_file.resolve())

    new_file = new_root / "src" / "a.py"
    new_file.parent.mkdir(parents=True)
    new_file.write_text(source_text, encoding="utf-8")
    leftover = str((tmp_path / "nowhere" / "leftover.py").resolve())

    storage = await open_db_connection(
        new_root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    intelligence = MagicMock()
    intelligence.summarize = AsyncMock(side_effect=AssertionError("llm"))
    try:
        for unit in parsed:
            unit.summary = "kept"
            unit.path = old_path
            unit.id = f"{old_path}:{unit.name}"
            await storage.upsert_unit(unit, vector=[0.0] * 384)
        await storage.upsert_unit(_unit(f"{leftover}:alpha", leftover, "unmatched"))

        result = await run_sync(
            IndexStack(storage, MultiParser(), intelligence),
            SyncOptions(root=new_root, path=None, index_all=True),
        )
        assert result.status == "success"
        intelligence.summarize.assert_not_awaited()
        paths = {
            row[0] for row in storage.conn.execute("SELECT path FROM units").fetchall()
        }
        assert "src/a.py" in paths
        assert leftover in paths
        assert old_path not in paths
        ids = {
            row[0]
            for row in storage.conn.execute("SELECT id FROM unit_embeddings").fetchall()
        }
        assert any(item.startswith("src/a.py:") for item in ids)
        assert await storage.paths_migration_done() is True

        later_path = str((tmp_path / "later-old" / "moved.py").resolve())
        (new_root / "later.py").write_text(source_text, encoding="utf-8")
        intelligence.summarize = AsyncMock(return_value="new")
        await storage.upsert_unit(
            _unit(f"{later_path}:alpha", later_path, parsed[0].code_hash)
        )
        await run_sync(
            IndexStack(storage, MultiParser(), intelligence),
            SyncOptions(root=new_root, path=None, index_all=True),
        )
        still = storage.conn.execute(
            "SELECT path FROM units WHERE path = ?",
            [later_path],
        ).fetchall()
        assert still == [(later_path,)]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_parser_error_does_not_set_mark(tmp_path: Path):
    storage = await open_db_connection(
        tmp_path / "t.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )

    class Boom:
        async def distill_file(self, file_path: str, *, stored_path: str | None = None):
            raise RuntimeError("boom")

    try:
        gone = str((tmp_path / "gone" / "a.py").resolve())
        await storage.upsert_unit(_unit(f"{gone}:alpha", gone, "h1"))
        with pytest.raises(RuntimeError, match="boom"):
            await migrate_absolute_paths(
                storage,
                Boom(),
                tmp_path / "proj",
                [tmp_path / "proj" / "a.py"],
            )
        assert await storage.paths_migration_done() is False
        left = storage.conn.execute("SELECT path FROM units").fetchone()
        assert left[0] == gone
    finally:
        await storage.close()
