from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_rag.services.indexing import IndexStack
from code_rag.services.sync import SyncOptions, run_sync
from code_rag.parsers.multi_parser import MultiParser
from code_rag.storage.db_connection import AccessMode, open_db_connection
from tests.embedder_stubs import StubEmbedder


@pytest.mark.asyncio
async def test_sync_stores_posix_path_relative_to_root(tmp_path: Path):
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
        result = await run_sync(
            IndexStack(storage, MultiParser(), intelligence),
            SyncOptions(root=root, path=None, index_all=True),
        )
        assert result.status == "success"
        rows = storage.conn.execute("SELECT id, path FROM units").fetchall()
        assert rows
        assert {row[1] for row in rows} == {"src/a.py"}
        assert all(row[0].startswith("src/a.py:") for row in rows)
        assert all(
            ":\\" not in row[0] and not row[0].startswith("src\\") for row in rows
        )
    finally:
        await storage.close()
