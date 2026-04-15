import pytest

from code_rag.core.interfaces import IParser, IStorage, IIntelligence
from code_rag.core.models import KnowledgeUnit, UnitKind


class TestInterfaces:
    def test_iparser_is_abstract(self):
        with pytest.raises(TypeError):
            IParser()
    
    def test_istorage_is_abstract(self):
        with pytest.raises(TypeError):
            IStorage()
    
    def test_iintelligence_is_abstract(self):
        with pytest.raises(TypeError):
            IIntelligence()
    
    @pytest.mark.asyncio
    async def test_iparser_implementation(self):
        class MockParser(IParser):
            async def distill_file(self, file_path: str):
                return [KnowledgeUnit(id="test", kind=UnitKind.FUNCTION, name="test", path="test.py", code_hash="abc")]
        parser = MockParser()
        result = await parser.distill_file("test.py")
        assert len(result) == 1