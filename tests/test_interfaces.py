import pytest
from typing import List, Optional
from code_rag.core.interfaces import IParser, IStorage, IIntelligence
from code_rag.core.models import KnowledgeUnit, Relation


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
            async def upsert_unit(self, unit: KnowledgeUnit):
                pass

            async def get_unit(self, unit_id: str):
                return None

            async def search_units(self, query: str, limit: int = 5):
                return []

            async def upsert_relation(self, relation: Relation):
                pass

            async def get_relations(self, unit_id: str, direction: str = "out"):
                return []

            async def set_dependency_path(self, lib_name: str, path: str) -> None:
                pass

            async def get_dependency_path(self, lib_name: str) -> Optional[str]:
                return None

            async def close(self) -> None:
                pass

        storage = MockStorage()
        assert isinstance(storage, IStorage)


class TestIIntelligence:
    """Tests for IIntelligence interface."""

    def test_iintelligence_subclass(self):
        """Test that subclass can implement IIntelligence."""

        class MockIntelligence(IIntelligence):
            async def summarize(self, code: str, unit_name: str) -> str:
                return "summary"

        intel = MockIntelligence()
        assert isinstance(intel, IIntelligence)
