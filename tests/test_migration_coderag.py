import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from code_rag.api.client import CodeRAG
from code_rag.storage.db_connection import AccessMode, open_db_connection
from code_rag.storage.duckdb_impl import PATHS_MIGRATED_KEY
from tests.embedder_stubs import StubEmbedder
from tests.migration_e2e_support import VECTOR, isolated_home_at, seed_parsed_units

PYTHON = "def alpha():\n    return 1\n"
STUB = StubEmbedder(dim=8, model_id="stub")


def _outside(tmp_path: Path, name: str) -> str:
    return str((tmp_path / "outside" / name).resolve())


async def _mark(db: Path) -> str | None:
    storage = await open_db_connection(
        db, STUB, mode=AccessMode.READ_ONLY, connect_timeout_seconds=0
    )
    try:
        row = storage.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            [PATHS_MIGRATED_KEY],
        ).fetchone()
        return None if row is None else row[0]
    finally:
        await storage.close()


async def _rows(db: Path) -> list[tuple]:
    storage = await open_db_connection(
        db, STUB, mode=AccessMode.READ_ONLY, connect_timeout_seconds=0
    )
    try:
        return storage.conn.execute(
            "SELECT path, summary, code_hash FROM units"
        ).fetchall()
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_changed_file_deletes_outside_absolute_path(tmp_path: Path):
    root = tmp_path / "proj"
    inside = root / "src" / "a.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("def alpha():\n    return 2\n", encoding="utf-8")
    seed_src = tmp_path / "seed.py"
    seed_src.write_text(PYTHON, encoding="utf-8")
    old = _outside(tmp_path, "a.py")
    db = root / ".coderag.db"
    home = tmp_path / "home"
    with isolated_home_at(home):
        await seed_parsed_units(db, STUB, [(seed_src, old, "kept")])
        with patch(
            "code_rag.services.factory.create_embedder",
            new=AsyncMock(return_value=STUB),
        ):
            async with CodeRAG(db=str(db), root=root) as rag:
                result = await rag.sync(index_all=True)
        assert result.status == "success"
        rows = await _rows(db)
        assert old not in {row[0] for row in rows}
        assert "src/a.py" in {row[0] for row in rows}


@pytest.mark.asyncio
async def test_missing_java_grammar_keeps_outside_java_path(tmp_path: Path):
    if importlib.util.find_spec("tree_sitter_java") is not None:
        pytest.fail("tree_sitter_java is installed; this test requires it to be absent")
    root = tmp_path / "proj"
    python_file = root / "src" / "a.py"
    java_file = root / "src" / "A.java"
    python_file.parent.mkdir(parents=True)
    python_file.write_text(PYTHON, encoding="utf-8")
    java_file.write_text("class A {}\n", encoding="utf-8")
    outside_py = tmp_path / "outside" / "a.py"
    outside_py.parent.mkdir(parents=True)
    outside_py.write_text(PYTHON, encoding="utf-8")
    old_py = str(outside_py.resolve())
    old_java = _outside(tmp_path, "A.java")
    db = root / ".coderag.db"
    home = tmp_path / "home"
    with isolated_home_at(home):
        await seed_parsed_units(db, STUB, [(outside_py, old_py, "kept")])
        seeded_hash = next(row[2] for row in await _rows(db) if row[0] == old_py)
        storage = await open_db_connection(
            db, STUB, mode=AccessMode.READ_WRITE, connect_timeout_seconds=0
        )
        try:
            from code_rag.core.models import KnowledgeUnit, UnitKind

            unit = KnowledgeUnit(
                id=f"{old_java}:A",
                name="A",
                kind=UnitKind.CLASS,
                path=old_java,
                code_hash="java-hash",
                summary="kept-java",
            )
            await storage.upsert_unit(unit, vector=list(VECTOR))
        finally:
            await storage.close()
        with patch(
            "code_rag.services.factory.create_embedder",
            new=AsyncMock(return_value=STUB),
        ):
            async with CodeRAG(db=str(db), root=root) as rag:
                await rag.sync(index_all=True)
                rows = await _rows(db)
                paths = {row[0] for row in rows}
                assert old_py not in paths
                assert ("src/a.py", "kept", seeded_hash) in rows
                assert old_java in paths
                assert await _mark(db) is None
                await rag.sync(index_all=True)
        storage = await open_db_connection(
            db, STUB, mode=AccessMode.READ_ONLY, connect_timeout_seconds=0
        )
        try:
            python_ids = [
                row[0]
                for row in storage.conn.execute(
                    "SELECT id FROM units WHERE path = ?", ["src/a.py"]
                ).fetchall()
            ]
        finally:
            await storage.close()
    assert python_ids
    assert all(item.startswith("src/a.py:") for item in python_ids)


@pytest.mark.asyncio
async def test_read_failure_keeps_outside_absolute_path(tmp_path: Path):
    root = tmp_path / "proj"
    inside = root / "src" / "a.py"
    inside.parent.mkdir(parents=True)
    inside.write_text(PYTHON, encoding="utf-8")
    outside = tmp_path / "outside" / "a.py"
    outside.parent.mkdir(parents=True)
    outside.write_text(PYTHON, encoding="utf-8")
    old = str(outside.resolve())
    db = root / ".coderag.db"
    home = tmp_path / "home"
    real_open = __import__("aiofiles").open

    def boom(path, *args, **kwargs):
        if Path(path).resolve() == inside.resolve():
            raise OSError("unreadable")
        return real_open(path, *args, **kwargs)

    with isolated_home_at(home):
        await seed_parsed_units(db, STUB, [(outside, old, "kept")])
        with patch("code_rag.parsers.tree_sitter.aiofiles.open", boom), patch(
            "code_rag.services.factory.create_embedder",
            new=AsyncMock(return_value=STUB),
        ):
            async with CodeRAG(db=str(db), root=root) as rag:
                await rag.sync(index_all=True)
        paths = {row[0] for row in await _rows(db)}
        assert old in paths
        assert await _mark(db) is None


@pytest.mark.asyncio
async def test_path_inside_root_keeps_summary_and_hash(tmp_path: Path):
    root = tmp_path / "proj"
    target = root / "src" / "a.py"
    other = root / "src" / "b.py"
    target.parent.mkdir(parents=True)
    target.write_text(PYTHON, encoding="utf-8")
    other.write_text("def beta():\n    return 1\n", encoding="utf-8")
    stored = str(target.resolve())
    db = root / ".coderag.db"
    home = tmp_path / "home"
    with isolated_home_at(home):
        await seed_parsed_units(db, STUB, [(target, stored, "kept")])
        seeded_hash = (await _rows(db))[0][2]
        target.write_text("def alpha():\n    return 9\n", encoding="utf-8")
        with patch(
            "code_rag.services.factory.create_embedder",
            new=AsyncMock(return_value=STUB),
        ):
            async with CodeRAG(db=str(db), root=root) as rag:
                result = await rag.sync(str(other))
        assert result.status == "success"
        rows = await _rows(db)
        assert ("src/a.py", "kept", seeded_hash) in rows


@pytest.mark.asyncio
async def test_equal_hashes_claim_exactly_one_file(tmp_path: Path):
    root = tmp_path / "proj"
    first = root / "src" / "a.py"
    second = root / "src" / "b.py"
    first.parent.mkdir(parents=True)
    first.write_text(PYTHON, encoding="utf-8")
    second.write_text(PYTHON, encoding="utf-8")
    old = _outside(tmp_path, "old.py")
    db = root / ".coderag.db"
    home = tmp_path / "home"
    with isolated_home_at(home):
        await seed_parsed_units(db, STUB, [(first, old, "kept")])
        with patch(
            "code_rag.services.factory.create_embedder",
            new=AsyncMock(return_value=STUB),
        ):
            async with CodeRAG(db=str(db), root=root) as rag:
                await rag.sync(index_all=True)
        rows = await _rows(db)
        kept = [row for row in rows if row[1] == "kept"]
        assert len(kept) == 1
        assert kept[0][0] in {"src/a.py", "src/b.py"}
        assert old not in {row[0] for row in rows}
        assert {row[0] for row in rows} >= {"src/a.py", "src/b.py"}


@pytest.mark.asyncio
async def test_second_sync_does_not_walk_root_for_migration(tmp_path: Path):
    root = tmp_path / "proj"
    source = root / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text(PYTHON, encoding="utf-8")
    db = root / ".coderag.db"
    home = tmp_path / "home"
    with isolated_home_at(home):
        await seed_parsed_units(db, STUB, [(source, str(source.resolve()), "kept")])
        with patch(
            "code_rag.services.factory.create_embedder",
            new=AsyncMock(return_value=STUB),
        ):
            async with CodeRAG(db=str(db), root=root) as rag:
                await rag.sync(index_all=True)
            assert await _mark(db) == "1"
            with patch("code_rag.services.sync._indexable_under") as walk:
                async with CodeRAG(db=str(db), root=root) as rag:
                    result = await rag.sync(str(source))
        assert result.status == "success"
        walk.assert_not_called()
