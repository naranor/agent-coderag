import pytest
from unittest.mock import MagicMock, AsyncMock

from code_rag.core.manager import CodeRAGManager
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.parsers.multi_parser import MultiParser
from code_rag.intelligence.distiller import Distiller
from code_rag.core.models import UnitKind, KnowledgeUnit
from tests.embedder_stubs import StubEmbedder


@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_knowledge.db")


@pytest.mark.asyncio
async def test_coderag_sync_and_search(temp_db):
    mock_embedder = StubEmbedder(dim=384)
    storage = await DuckDBStorage.open(temp_db, mock_embedder)

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

    manager = CodeRAGManager(storage, mock_parser, mock_distiller)
    await manager.sync_file("test_file.py", force_distill=True)
    res = storage.conn.execute("SELECT name, summary FROM units").fetchall()
    assert len(res) == 1
    assert res[0][0] == "test_func"
    assert res[0][1] == "This is a test function for RAG."
    search_results = await manager.search("test function", limit=1)
    assert len(search_results) == 1
    assert search_results[0].name == "test_func"
    await manager.close()
