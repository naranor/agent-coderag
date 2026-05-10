import pytest
from unittest.mock import AsyncMock, patch
from code_rag.parsers.multi_parser import MultiParser


class TestMultiParser:
    """Tests for MultiParser class."""

    @pytest.mark.asyncio
    async def test_multi_parser_delegation(self):
        parser = MultiParser()

        with patch.object(
            parser.tree_sitter_parser, "distill_file", new_callable=AsyncMock
        ) as mock_distill:
            mock_distill.return_value = []
            await parser.distill_file("test.py")
            mock_distill.assert_called_once_with("test.py")

            mock_distill.reset_mock()
            await parser.distill_file("test.js")
            mock_distill.assert_called_once_with("test.js")

    @pytest.mark.asyncio
    async def test_multi_parser_unknown_extension(self):
        parser = MultiParser()
        result = await parser.distill_file("test.unknown_ext_xyz")
        assert result == []

    def test_multi_parser_init(self):
        parser = MultiParser()
        assert parser.tree_sitter_parser is not None
