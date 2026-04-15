import pytest
from abc import ABC

from code_rag.core.interfaces import IParser, IStorage, IIntelligence
from code_rag.core.models import KnowledgeUnit, UnitKind


class TestIParser:
    def test_iparser_is_abstract(self):
        with pytest.raises(TypeError):
            IParser()
    
    @pytest.mark.asyncio
    async def test_iparser_subclass(self):
        class MockParser(IParser):
            async def distill_file(self, file_path: str):
                return [KnowledgeUnit(id="test", kind=UnitKind.FUNCTION, name="test", path="test.py", code_hash="abc")]
        
        parser = MockParser()
        result = await parser.distill_file("test.py")
        assert len(result) == 1


class TestIStorage:
    def test_istorage_is_abstract(self):
        with pytest.raises(TypeError):
            IStorage()
    
    @pytest.mark.asyncio
    async def test_istorage_subclass(self):
        class MockStorage(IStorage):
            async def upsert_unit(self, unit: KnowledgeUnit):
                pass
            async def get_unit(self, unit_id: str):
                return None
            async def search_units(self, query: str, limit: int = 5):
                return []
        
        storage = MockStorage()
        assert storage is not None


class TestIIntelligence:
    def test_iintelligence_is_abstract(self):
        with pytest.raises(TypeError):
            IIntelligence()
    
    @pytest.mark.asyncio
    async def test_iintelligence_subclass(self):
        class MockIntelligence(IIntelligence):
            async def summarize(self, code: str, unit_name: str):
                return "Mock summary"
        
        intelligence = MockIntelligence()
        result = await intelligence.summarize("def test(): pass", "test")
        assert result == "Mock summary"