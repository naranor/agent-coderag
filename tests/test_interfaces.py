import pytest
from typing import List, Optional
from code_rag.core.interfaces import IParser, IStorage, IIntelligence
from code_rag.core.models import KnowledgeUnit, Relation, UnitKind, RelationType


class TestIParser:
    """Tests for IParser interface."""

    def test_iparser_subclass(self):
        """Test that subclass can implement IParser."""

        class MockParser(IParser):
            async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
                return []

        parser = MockParser()
        assert isinstance(parser, IParser)


class TestIStorage:
    """Tests for IStorage interface."""

    @pytest.mark.asyncio
    async def test_istorage_subclass(self):
        """Test that subclass can implement IStorage."""

        class MockStorage(IStorage):
            def __init__(self):
                self.units = {}
                self.relations = []
                self.deps = {}

            async def upsert_unit(self, unit: KnowledgeUnit):
                self.units[unit.id] = unit

            async def get_unit(self, unit_id: str):
                return self.units.get(unit_id)

            async def search_units(self, query: str, limit: int = 5):
                return list(self.units.values())[:limit]

            async def upsert_relation(self, relation: Relation):
                self.relations.append(relation)

            async def get_relations(self, unit_id: str, direction: str = "out"):
                return self.relations

            async def set_dependency_path(self, lib_name: str, path: str) -> None:
                self.deps[lib_name] = path

            async def get_dependency_path(self, lib_name: str) -> Optional[str]:
                return self.deps.get(lib_name)

            async def delete_stale_units(
                self, file_path: str, current_unit_ids: List[str]
            ) -> None:
                pass

            async def close(self) -> None:
                pass

        storage = MockStorage()
        assert isinstance(storage, IStorage)

        # Exercise methods to satisfy coverage (VDP Phase 3)
        unit = KnowledgeUnit(
            id="u1", name="n1", kind=UnitKind.FUNCTION, path="p1", code_hash="h1"
        )
        await storage.upsert_unit(unit)
        assert await storage.get_unit("u1") == unit
        assert len(await storage.search_units("test")) == 1

        rel = Relation(from_id="u1", to_id="u2", type=RelationType.CALLS)
        await storage.upsert_relation(rel)
        rels = await storage.get_relations("u1")
        assert len(rels) == 1

        await storage.set_dependency_path("lib", "path")
        assert await storage.get_dependency_path("lib") == "path"

        await storage.delete_stale_units("p1", ["u1"])
        await storage.close()


class TestIIntelligence:
    """Tests for IIntelligence interface."""

    def test_iintelligence_subclass(self):
        """Test that subclass can implement IIntelligence."""

        class MockIntelligence(IIntelligence):
            async def summarize(self, code: str, unit_name: str) -> str:
                return "summary"

        intel = MockIntelligence()
        assert isinstance(intel, IIntelligence)
