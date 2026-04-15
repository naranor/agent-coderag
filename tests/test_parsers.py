import pytest
from code_rag.parsers.ast_index import AstIndexParser


class TestAstIndexParser:
    @pytest.mark.asyncio
    async def test_distill_file_returns_list(self):
        parser = AstIndexParser()
        result = await parser.distill_file("code_rag/core/models.py")
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_distill_file_valid_units(self):
        parser = AstIndexParser()
        result = await parser.distill_file("code_rag/core/models.py")
        assert len(result) > 0
        for unit in result:
            assert hasattr(unit, 'id')
            assert hasattr(unit, 'name')
    
    @pytest.mark.asyncio
    async def test_distill_nonexistent(self):
        parser = AstIndexParser()
        result = await parser.distill_file("nonexistent.py")
        assert result == []