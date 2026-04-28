import pytest
from unittest.mock import AsyncMock, patch
from code_rag.parsers.multi_parser import MultiParser

class TestMultiParser:
    """Tests for MultiParser class."""

    @pytest.mark.asyncio
    async def test_multi_parser_delegation(self):
        parser = MultiParser()
        
        with patch.object(parser.parsers[".py"], 'distill_file', new_callable=AsyncMock) as mock_py:
            await parser.distill_file("test.py")
            mock_py.assert_called_once_with("test.py")
            
        if ".java" in parser.parsers:
            with patch.object(parser.parsers[".java"], 'distill_file', new_callable=AsyncMock) as mock_java:
                await parser.distill_file("Test.java")
                mock_java.assert_called_once_with("Test.java")

    @pytest.mark.asyncio
    async def test_multi_parser_unknown_extension(self):
        parser = MultiParser()
        result = await parser.distill_file("test.txt")
        assert result == []

    def test_multi_parser_init(self):
        parser = MultiParser()
        assert ".py" in parser.parsers
        # .java might or might not be there depending on javalang installation
