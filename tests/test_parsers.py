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
            assert hasattr(unit, 'path')
    
    @pytest.mark.asyncio
    async def test_distill_file_nonexistent(self):
        parser = AstIndexParser()
        result = await parser.distill_file("nonexistent_file.py")
        assert result == []
    
    def test_parser_init(self):
        parser = AstIndexParser()
        assert parser is not None