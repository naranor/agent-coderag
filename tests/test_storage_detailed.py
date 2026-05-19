import pytest
import duckdb
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.core.models import KnowledgeUnit, UnitKind, Relation, RelationType


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.db"
    return DuckDBStorage(str(db_path))


class TestStorageDetailed:
    """Detailed tests for DuckDBStorage to increase coverage."""

    @pytest.mark.asyncio
    async def test_delete_stale_units(self, storage):
        """Test garbage collection of units."""
        u1 = KnowledgeUnit(
            id="f1:u1", name="u1", kind=UnitKind.FUNCTION, path="f1", code_hash="h1"
        )
        u2 = KnowledgeUnit(
            id="f1:u2", name="u2", kind=UnitKind.FUNCTION, path="f1", code_hash="h2"
        )
        await storage.upsert_unit(u1)
        await storage.upsert_unit(u2)

        # Confirm 2 units
        res = storage.conn.execute("SELECT count(*) FROM units").fetchone()
        assert res[0] == 2

        # Delete u2 by omitting it from current_unit_ids
        await storage.delete_stale_units("f1", ["f1:u1"])

        res = storage.conn.execute("SELECT id FROM units").fetchall()
        assert len(res) == 1
        assert res[0][0] == "f1:u1"

    @pytest.mark.asyncio
    async def test_search_units_batch_relations(self, storage):
        """Test search_units correctly fetches relations in batch."""
        u1 = KnowledgeUnit(
            id="u1", name="u1", kind=UnitKind.FUNCTION, path="p1", code_hash="h1"
        )
        rel = Relation(from_id="u1", to_id="u2", type=RelationType.CALLS)
        u1.relations = [rel]

        await storage.upsert_unit(u1)

        # Search for it
        results = await storage.search_units("u1")
        assert len(results) == 1
        assert len(results[0].relations) == 1
        assert results[0].relations[0].to_id == "u2"

    @pytest.mark.asyncio
    async def test_dependency_path_caching(self, storage):
        """Test caching and retrieval of dependency paths."""
        await storage.set_dependency_path("lib-x", "/path/to/lib-x.jar")
        path = await storage.get_dependency_path("lib-x")
        assert path == "/path/to/lib-x.jar"

        # Missing
        assert await storage.get_dependency_path("missing") is None

    @pytest.mark.asyncio
    async def test_close_storage(self, storage):
        """Test closing connection."""
        await storage.close()
        # Connection should be unusable
        with pytest.raises(duckdb.ConnectionException):
            storage.conn.execute("SELECT 1")
