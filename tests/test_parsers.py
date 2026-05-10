import pytest

from code_rag.parsers.ast_index import AstIndexParser


class TestAstIndexParser:
    """Tests for AstIndexParser."""

    @pytest.mark.asyncio
    async def test_distill_file_returns_list(self):
        """Test that distill_file returns a list."""
        parser = AstIndexParser()
        result = await parser.distill_file("code_rag/core/models.py")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_distill_file_valid_units(self):
        """Test that distill_file returns valid KnowledgeUnits."""
        parser = AstIndexParser()
        result = await parser.distill_file("code_rag/core/models.py")

        # Check that we have units
        assert len(result) > 0

        # Check unit structure
        for unit in result:
            assert hasattr(unit, "id")
            assert hasattr(unit, "name")
            assert hasattr(unit, "path")
            assert hasattr(unit, "kind")

    @pytest.mark.asyncio
    async def test_distill_file_nonexistent(self):
        """Test handling of nonexistent file."""
        parser = AstIndexParser()
        result = await parser.distill_file("nonexistent_file.py")
        assert result == []

    def test_parser_init(self):
        """Test parser initialization."""
        parser = AstIndexParser()
        assert parser is not None
