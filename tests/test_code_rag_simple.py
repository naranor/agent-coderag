import pytest
from unittest.mock import AsyncMock, MagicMock

from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.intelligence.distiller import Distiller
from code_rag.parsers.multi_parser import MultiParser
from code_rag.services.indexing import IndexStack, sync_file
from code_rag.services.search import run_search
from code_rag.storage.db_connection import AccessMode, open_db_connection
from tests.embedder_stubs import StubEmbedder


@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_knowledge.db")


@pytest.mark.asyncio
async def test_coderag_sync_and_search(temp_db):
    mock_embedder = StubEmbedder(dim=384)
    storage = await open_db_connection(
        temp_db, mock_embedder, mode=AccessMode.READ_WRITE, connect_timeout_seconds=0
    )

    mock_parser = MagicMock(spec=MultiParser)
    test_unit = KnowledgeUnit(
        id="test_file.py:test_func",
        kind=UnitKind.FUNCTION,
        name="test_func",
        path="test_file.py",
        code_hash="abc123hash",
        metadata={"raw_code": "def test_func(): pass"},
    )
    mock_parser.distill_file = AsyncMock(return_value=[test_unit])

    mock_distiller = MagicMock(spec=Distiller)
    mock_distiller.summarize = AsyncMock(
        return_value="This is a test function for RAG."
    )

    await sync_file(
        IndexStack(storage, mock_parser, mock_distiller),
        "test_file.py",
        force_distill=True,
    )
    res = storage.conn.execute("SELECT name, summary FROM units").fetchall()
    assert len(res) == 1
    assert res[0][0] == "test_func"
    assert res[0][1] == "This is a test function for RAG."
    search_results = await run_search(storage, "test function", limit=1)
    assert len(search_results) == 1
    assert search_results[0].name == "test_func"
    await storage.close()
