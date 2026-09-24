from pathlib import Path

import pytest

from code_rag.storage.db_connection import AccessMode, open_db_connection
from code_rag.storage.duckdb_impl import PATHS_MIGRATED_KEY
from tests.embedder_stubs import StubEmbedder
from tests.migration_e2e_support import (
    MODEL_ID,
    EmbeddingServer,
    isolated_home_at,
    run_cli,
    seed_parsed_units,
    write_embedding_config,
)

SOURCE = "def alpha():\n    return 1\n"
RECOMMEND = "Run sync --all to reindex the project after path migration."


def _paths(storage) -> set[str]:
    return {row[0] for row in storage.conn.execute("SELECT path FROM units").fetchall()}


@pytest.mark.asyncio
async def test_cli_sync_all_moves_tree(tmp_path: Path):
    project = tmp_path / "proj"
    source = project / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text(SOURCE, encoding="utf-8")
    old = str((tmp_path / "old" / "src" / "a.py").resolve())
    home = tmp_path / "home"
    server = EmbeddingServer()
    server.start()
    try:
        with isolated_home_at(home):
            write_embedding_config(home, server.base_url)
            await seed_parsed_units(
                project / ".coderag.db",
                StubEmbedder(dim=8, model_id=MODEL_ID),
                [(source, old, "kept")],
            )
            result = run_cli(["sync", "--all"], cwd=project, home=home)
        assert result.returncode == 0, result.stderr
        storage = await open_db_connection(
            project / ".coderag.db",
            StubEmbedder(dim=8, model_id=MODEL_ID),
            mode=AccessMode.READ_ONLY,
            connect_timeout_seconds=0,
        )
        try:
            assert _paths(storage) == {"src/a.py"}
            ids = storage.conn.execute("SELECT id FROM unit_embeddings").fetchall()
            assert ids and all(row[0].startswith("src/a.py:") for row in ids)
            summary = storage.conn.execute(
                "SELECT summary FROM units WHERE path = ?", ["src/a.py"]
            ).fetchone()
            assert summary == ("kept",)
        finally:
            await storage.close()
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_cli_search_before_and_after_migration(tmp_path: Path):
    project = tmp_path / "proj"
    source = project / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text(SOURCE, encoding="utf-8")
    old = str((tmp_path / "old" / "src" / "a.py").resolve())
    home = tmp_path / "home"
    server = EmbeddingServer()
    server.start()
    try:
        with isolated_home_at(home):
            write_embedding_config(home, server.base_url)
            await seed_parsed_units(
                project / ".coderag.db",
                StubEmbedder(dim=8, model_id=MODEL_ID),
                [(source, old, "kept")],
            )
            before = run_cli(["search", "alpha"], cwd=project, home=home)
            assert before.returncode == 0, before.stderr
            assert old in before.stdout
            moved = run_cli(["sync", "--all"], cwd=project, home=home)
            assert moved.returncode == 0, moved.stderr
            after = run_cli(["search", "alpha"], cwd=project, home=home)
            assert after.returncode == 0, after.stderr
            assert str((project / "src" / "a.py").resolve()) in after.stdout
            relative = run_cli(
                ["search", "--relative-paths", "alpha"], cwd=project, home=home
            )
            assert relative.returncode == 0, relative.stderr
            resolved = str((project / "src" / "a.py").resolve())
            assert "src/a.py" in relative.stdout
            assert resolved not in relative.stdout
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_cli_partial_sync_recommends_full_sync(tmp_path: Path):
    project = tmp_path / "proj"
    source = project / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text(SOURCE, encoding="utf-8")
    old = str((tmp_path / "old" / "a.py").resolve())
    home = tmp_path / "home"
    server = EmbeddingServer()
    server.start()
    try:
        with isolated_home_at(home):
            write_embedding_config(home, server.base_url)
            await seed_parsed_units(
                project / ".coderag.db",
                StubEmbedder(dim=8, model_id=MODEL_ID),
                [(source, old, "kept")],
            )
            first = run_cli(["sync", str(source)], cwd=project, home=home)
            assert first.returncode == 0, first.stderr
            assert RECOMMEND in first.stderr
            storage = await open_db_connection(
                project / ".coderag.db",
                StubEmbedder(dim=8, model_id=MODEL_ID),
                mode=AccessMode.READ_ONLY,
                connect_timeout_seconds=0,
            )
            try:
                mark = storage.conn.execute(
                    "SELECT value FROM index_meta WHERE key = ?",
                    [PATHS_MIGRATED_KEY],
                ).fetchone()
                assert mark == ("1",)
            finally:
                await storage.close()
            second = run_cli(["sync", str(source)], cwd=project, home=home)
            assert second.returncode == 0, second.stderr
            assert RECOMMEND not in second.stderr
    finally:
        server.stop()
