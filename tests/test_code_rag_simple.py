import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from code_rag.core.manager import CodeRAGManager
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.parsers.multi_parser import MultiParser
from code_rag.intelligence.embedder import Embedder
from code_rag.intelligence.distiller import Distiller
from code_rag.core.models import UnitKind, KnowledgeUnit


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_knowledge.db"
    return str(db_path)


@pytest.mark.asyncio
async def test_coderag_sync_and_search(temp_db):
    """
    Verifies the full pipeline: Parse -> Distill -> Store -> Search.
    """
    # 1. Setup Mock Embedder (returns random vectors instead of running ONNX)
    mock_embedder = MagicMock(spec=Embedder)
    # Generate a random 384-dim vector for any input
    mock_embedder.embed.side_effect = lambda texts: np.random.rand(
        len(texts), 384
    ).astype(np.float32)

    # 2. Setup Storage
    storage = DuckDBStorage(temp_db, embedder=mock_embedder)

    # 3. Setup Mock Parser (simulates ast-index output)
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

    # 4. Setup Mock Distiller (simulates LLM summary)
    mock_distiller = MagicMock(spec=Distiller)
    mock_distiller.summarize = AsyncMock(
        return_value="This is a test function for RAG."
    )

    # 5. Initialize Manager
    manager = CodeRAGManager(storage, mock_parser, mock_distiller)

    # 6. RUN SYNC
    await manager.sync_file("test_file.py", force_distill=True)

    # 7. VERIFY STORAGE
    # Check if unit exists in DuckDB
    res = storage.conn.execute("SELECT name, summary FROM units").fetchall()
    assert len(res) == 1
    assert res[0][0] == "test_func"
    assert res[0][1] == "This is a test function for RAG."

    # 8. RUN SEARCH
    search_results = await manager.search("test function", limit=1)
    assert len(search_results) == 1
    assert search_results[0].name == "test_func"

    print("\n[SUCCESS] CodeRAG Integration Test Passed!")


if __name__ == "__main__":
    # For manual running
    asyncio.run(test_coderag_sync_and_search("manual_test.db"))
