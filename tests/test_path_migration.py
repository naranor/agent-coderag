from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.core.models import KnowledgeUnit, Relation, RelationType, UnitKind
from code_rag.parsers.multi_parser import MultiParser
from code_rag.parsers.tree_sitter import GrammarNotFoundError
from code_rag.services.indexing import IndexStack
from code_rag.services.path_migration import (
    claim_relative_paths,
    migrate_absolute_paths,
)
from code_rag.services.sync import SyncOptions, _indexable_under, run_sync
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
        assert leftover not in paths
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
        async def distill_file(
            self,
            file_path: str,
            *,
            stored_path: str | None = None,
            raise_on_failure: bool = False,
        ):
            raise RuntimeError("boom")

    try:
        gone = str((tmp_path / "gone" / "a.py").resolve())
        await storage.upsert_unit(_unit(f"{gone}:alpha", gone, "h1"))
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


class _SplitParser:
    def __init__(self, python_units):
        self.python_units = python_units

    async def distill_file(
        self,
        file_path: str,
        *,
        stored_path: str | None = None,
        raise_on_failure: bool = False,
    ):
        if str(file_path).endswith(".java"):
            raise GrammarNotFoundError("java")
        return list(self.python_units)


@pytest.mark.asyncio
async def test_missing_java_grammar_keeps_python_commit(tmp_path: Path):
    root = tmp_path / "proj"
    python_file = root / "src" / "a.py"
    java_file = root / "src" / "A.java"
    python_file.parent.mkdir(parents=True)
    python_file.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    java_file.write_text("class A {}\n", encoding="utf-8")

    parsed = await MultiParser().distill_file(str(python_file.resolve()))
    assert parsed
    old_python = str((tmp_path / "old" / "src" / "a.py").resolve())
    old_java = str((tmp_path / "old" / "src" / "A.java").resolve())

    storage = await open_db_connection(
        root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        for unit in parsed:
            unit.summary = "kept"
            unit.path = old_python
            unit.id = f"{old_python}:{unit.name}"
            await storage.upsert_unit(unit, vector=[0.0] * 384)
        await storage.upsert_unit(_unit(f"{old_java}:A", old_java, "java-hash"))

        await migrate_absolute_paths(
            storage,
            _SplitParser(parsed),
            root,
            [python_file, java_file],
        )
        paths = {
            row[0] for row in storage.conn.execute("SELECT path FROM units").fetchall()
        }
        assert "src/a.py" in paths
        assert old_java in paths
        assert await storage.paths_migration_done() is False

        await migrate_absolute_paths(
            storage,
            _SplitParser(parsed),
            root,
            [python_file, java_file],
        )
        python_ids = storage.conn.execute(
            "SELECT id FROM units WHERE path = ?",
            ["src/a.py"],
        ).fetchall()
        assert python_ids
        assert all(row[0].startswith("src/a.py:") for row in python_ids)
        java_left = storage.conn.execute(
            "SELECT path FROM units WHERE path = ?",
            [old_java],
        ).fetchall()
        assert java_left == [(old_java,)]
        assert await storage.paths_migration_done() is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_changed_python_absolute_path_is_deleted_while_java_is_kept(
    tmp_path: Path,
):
    """A moved file whose hash changed leaves an orphan. Same-suffix parse success deletes it."""
    root = tmp_path / "proj"
    python_file = root / "src" / "a.py"
    java_file = root / "src" / "A.java"
    python_file.parent.mkdir(parents=True)
    python_file.write_text("def alpha():\n    return 2\n", encoding="utf-8")
    java_file.write_text("class A {}\n", encoding="utf-8")
    old_python = str((tmp_path / "old" / "src" / "a.py").resolve())
    old_java = str((tmp_path / "old" / "src" / "A.java").resolve())

    class Parser:
        async def distill_file(
            self,
            file_path: str,
            *,
            stored_path: str | None = None,
            raise_on_failure: bool = False,
        ):
            if str(file_path).endswith(".java"):
                raise GrammarNotFoundError("java")
            return []

    storage = await open_db_connection(
        root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        py = _unit(f"{old_python}:alpha", old_python, "stale-hash")
        jv = _unit(f"{old_java}:A", old_java, "java-hash")
        jv.relations = [Relation(from_id=jv.id, to_id=py.id, type=RelationType.CALLS)]
        await storage.upsert_unit(py, vector=[0.0] * 384)
        await storage.upsert_unit(jv, vector=[0.0] * 384)
        await migrate_absolute_paths(storage, Parser(), root, [python_file, java_file])
        paths = {
            row[0] for row in storage.conn.execute("SELECT path FROM units").fetchall()
        }
        assert old_python not in paths
        assert old_java in paths
        embeds = storage.conn.execute("SELECT id FROM unit_embeddings").fetchall()
        assert [row[0] for row in embeds] == [jv.id]
        rels = storage.conn.execute("SELECT from_id, to_id FROM relations").fetchall()
        assert rels == []
        assert await storage.paths_migration_done() is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_partial_sync_without_absolute_paths_does_not_recommend_sync_all(
    tmp_path: Path, capsys
):
    root = tmp_path / "proj"
    source = root / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    storage = await open_db_connection(
        root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    intelligence = MagicMock()
    intelligence.summarize = AsyncMock(return_value="kept")
    try:
        await storage.upsert_unit(_unit("src/a.py:alpha", "src/a.py", "stale"))
        await run_sync(
            IndexStack(storage, MultiParser(), intelligence),
            SyncOptions(root=root, path=str(source), index_all=False),
        )
        assert "sync --all" not in capsys.readouterr().err
        assert await storage.paths_migration_done() is True
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_partial_sync_recommends_sync_all_after_migration(tmp_path: Path, capsys):
    root = tmp_path / "proj"
    source = root / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    old = str((tmp_path / "old" / "a.py").resolve())
    storage = await open_db_connection(
        root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    intelligence = MagicMock()
    intelligence.summarize = AsyncMock(return_value="kept")
    try:
        await storage.upsert_unit(_unit(f"{old}:alpha", old, "stale"))
        await run_sync(
            IndexStack(storage, MultiParser(), intelligence),
            SyncOptions(root=root, path=str(source), index_all=False),
        )
        assert "sync --all" in capsys.readouterr().err
        capsys.readouterr()
        await run_sync(
            IndexStack(storage, MultiParser(), intelligence),
            SyncOptions(root=root, path=None, index_all=True),
        )
        assert "sync --all" not in capsys.readouterr().err
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_swallowed_parse_error_keeps_absolute_path_during_migration(
    tmp_path: Path,
):
    root = tmp_path / "proj"
    good_file = root / "src" / "a.py"
    missing_file = root / "src" / "b.py"
    good_file.parent.mkdir(parents=True)
    good_file.write_text("def alpha():\n    return 1\n", encoding="utf-8")

    parsed = await MultiParser().distill_file(str(good_file.resolve()))
    assert parsed
    old_good = str((tmp_path / "old" / "src" / "a.py").resolve())
    old_bad = str((tmp_path / "old" / "src" / "b.py").resolve())

    storage = await open_db_connection(
        root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        for unit in parsed:
            unit.summary = "kept"
            unit.path = old_good
            unit.id = f"{old_good}:{unit.name}"
            await storage.upsert_unit(unit, vector=[0.0] * 384)
        await storage.upsert_unit(_unit(f"{old_bad}:beta", old_bad, "orphan-hash"))

        await migrate_absolute_paths(
            storage,
            MultiParser(),
            root,
            [good_file, missing_file],
        )
        paths = {
            row[0] for row in storage.conn.execute("SELECT path FROM units").fetchall()
        }
        assert "src/a.py" in paths
        assert old_bad in paths
        assert old_good not in paths
        assert await storage.paths_migration_done() is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_sync_skips_migration_walk_when_mark_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "proj"
    source = root / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    storage = await open_db_connection(
        root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    intelligence = MagicMock()
    intelligence.summarize = AsyncMock(return_value="kept")
    calls: list[tuple] = []

    def counting(walk_root, ignore_spec, *, project_root):
        calls.append((walk_root, project_root))
        return _indexable_under(walk_root, ignore_spec, project_root=project_root)

    monkeypatch.setattr("code_rag.services.sync._indexable_under", counting)
    try:
        await storage.mark_paths_migrated()
        await run_sync(
            IndexStack(storage, MultiParser(), intelligence),
            SyncOptions(root=root, path=str(source), index_all=False),
        )
        assert calls == []
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_sync_indexes_when_migration_raises(tmp_path: Path):
    root = tmp_path / "proj"
    source = root / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    storage = await open_db_connection(
        root / ".coderag.db",
        StubEmbedder(),
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    intelligence = MagicMock()
    intelligence.summarize = AsyncMock(return_value="kept")

    async def boom(*_args, **_kwargs):
        raise RuntimeError("migration failed")

    try:
        with patch(
            "code_rag.services.sync.migrate_absolute_paths",
            new=boom,
        ):
            result = await run_sync(
                IndexStack(storage, MultiParser(), intelligence),
                SyncOptions(root=root, path=None, index_all=True),
            )
        assert result.status == "success"
        rows = storage.conn.execute("SELECT path FROM units").fetchall()
        assert {row[0] for row in rows} == {"src/a.py"}
    finally:
        await storage.close()
