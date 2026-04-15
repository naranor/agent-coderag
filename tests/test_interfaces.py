import pytest
from abc import ABC

from code_rag.core.interfaces import IParser, IStorage, IIntelligence
from code_rag.core.models import KnowledgeUnit, UnitKind


class TestIParser:
    """Tests for IParser interface."""
    
    def test_iparser_is_abstract(self):
        """Test that IParser is an abstract class."""
        with pytest.raises(TypeError):
            IParser()
    
    @pytest.mark.asyncio
    async def test_iparser_subclass(self):
        """Test that subclass can implement IParser."""
        class MockParser(IParser):
            async def distill_file(self, file_path: str):
                return [KnowledgeUnit(
                    id="test",
                    kind=UnitKind.FUNCTION,
                    name="test",
                    path="test.py",
                    code_hash="abc"
                )]
        
        parser = MockParser()
        result = await parser.distill_file("test.py")
        assert len(result) == 1


class TestIStorage:
    """Tests for IStorage interface."""
    
    def test_istorage_is_abstract(self):
        """Test that IStorage is an abstract class."""
        with pytest.raises(TypeError):
            IStorage()
    
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
        
        storage = MockStorage()
        assert storage is not None


class TestIIntelligence:
    """Tests for IIntelligence interface."""
    
    def test_iintelligence_is_abstract(self):
        """Test that IIntelligence is an abstract class."""
        with pytest.raises(TypeError):
            IIntelligence()
    
    @pytest.mark.asyncio
    async def test_iintelligence_subclass(self):
        """Test that subclass can implement IIntelligence."""
        class MockIntelligence(IIntelligence):
            async def summarize(self, code: str, unit_name: str):
                return "Mock summary"
        
        intelligence = MockIntelligence()
        result = await intelligence.summarize("def test(): pass", "test")
        assert result == "Mock summary"