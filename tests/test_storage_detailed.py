import pytest
import pytest_asyncio
import duckdb
from code_rag.storage.db_connection import AccessMode, open_db_connection
from code_rag.core.models import KnowledgeUnit, UnitKind, Relation, RelationType
from tests.embedder_stubs import StubEmbedder


@pytest_asyncio.fixture
async def storage(tmp_path):
    stub = StubEmbedder()
    store = await open_db_connection(
        tmp_path / "test.db",
        stub,
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    yield store
    await store.close()


class TestStorageDetailed:
    """Detailed tests for DuckDBStorage to increase coverage."""

    @pytest.mark.asyncio
    async def test_delete_stale_units(self, storage):
        """Test garbage collection of units."""
        vec = [0.0] * 384
        u1 = KnowledgeUnit(
            id="f1:u1", name="u1", kind=UnitKind.FUNCTION, path="f1", code_hash="h1"
        )
        u2 = KnowledgeUnit(
            id="f1:u2", name="u2", kind=UnitKind.FUNCTION, path="f1", code_hash="h2"
        )
        await storage.upsert_unit(u1, vector=vec)
        await storage.upsert_unit(u2, vector=vec)

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
        vec = [0.0] * 384
        u1 = KnowledgeUnit(
            id="u1", name="u1", kind=UnitKind.FUNCTION, path="p1", code_hash="h1"
        )
        rel = Relation(from_id="u1", to_id="u2", type=RelationType.CALLS)
        u1.relations = [rel]

        await storage.upsert_unit(u1, vector=vec)

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

    @pytest.mark.asyncio
    async def test_get_relations_inbound(self, storage):
        rel = Relation(from_id="src", to_id="dst", type=RelationType.CALLS)
        await storage.upsert_relation(rel)
        inbound = await storage.get_relations("dst", direction="in")
        assert len(inbound) == 1
        assert inbound[0].from_id == "src"

    @pytest.mark.asyncio
    async def test_map_row_empty_tags_and_metadata(self, storage):
        unit = KnowledgeUnit(
            id="empty",
            name="empty",
            kind=UnitKind.FUNCTION,
            path="p.py",
            code_hash="h",
            tags=[],
            metadata={},
        )
        await storage.upsert_unit(unit)
        storage.conn.execute(
            "UPDATE units SET tags = NULL, metadata = NULL WHERE id = 'empty'"
        )
        got = await storage.get_unit("empty")
        assert got is not None
        assert got.tags == []
        assert got.metadata == {}
