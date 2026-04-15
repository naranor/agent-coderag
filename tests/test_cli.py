import pytest
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch, AsyncMock
import argparse

from code_rag.entry import cli


class TestCLIHelpers:
    def test_should_index_py_file(self):
        from pathlib import Path
        path = Path("test.py")
        assert cli.should_index(path) is True
    
    def test_should_index_non_py_file(self):
        from pathlib import Path
        path = Path("test.txt")
        assert cli.should_index(path) is False
    
    def test_should_index_excluded_paths(self):
        from pathlib import Path
        assert cli.should_index(Path("tests/test.py")) is False
        assert cli.should_index(Path("venv/lib.py")) is False


class TestCLISearch:
    @pytest.mark.asyncio
    async def test_search_cmd_with_results(self):
        from code_rag.core.models import KnowledgeUnit, UnitKind
        
        mock_manager = MagicMock()
        mock_manager.search = AsyncMock(return_value=[
            KnowledgeUnit(id="test.py:test", kind=UnitKind.FUNCTION, name="test", path="test.py", summary="Test", code_hash="abc")
        ])
        
        with patch('code_rag.entry.cli.get_manager', return_value=mock_manager):
            args = argparse.Namespace(db="test.db", onnx=None, verbose=False, json=False, query="test", limit=5)
            old_stdout = sys.stdout
            sys.stdout = captured = StringIO()
            try:
                await cli.search_cmd(args)
            finally:
                sys.stdout = old_stdout
            assert "test | test.py" in captured.getvalue()
    
    @pytest.mark.asyncio
    async def test_search_cmd_no_results(self):
        mock_manager = MagicMock()
        mock_manager.search = AsyncMock(return_value=[])
        
        with patch('code_rag.entry.cli.get_manager', return_value=mock_manager):
            args = argparse.Namespace(db="test.db", onnx=None, verbose=False, json=False, query="nonexistent", limit=5)
            old_stdout = sys.stdout
            sys.stdout = captured = StringIO()
            try:
                await cli.search_cmd(args)
            finally:
                sys.stdout = old_stdout
            assert "No results" in captured.getvalue()


class TestCLIApi:
    @pytest.mark.asyncio
    async def test_api_cmd_success(self):
        with patch('code_rag.entry.cli.extract_library_api', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = {"functions": ["func1"]}
            args = argparse.Namespace(library="pydantic", json=False)
            old_stdout = sys.stdout
            sys.stdout = captured = StringIO()
            try:
                await cli.api_cmd(args)
            finally:
                sys.stdout = old_stdout
            assert "func1" in captured.getvalue()


class TestCLIConfig:
    def test_config_cmd_update(self):
        with patch('code_rag.entry.cli.DistillerConfig') as mock_config_cls:
            mock_config = MagicMock()
            mock_config_cls.load.return_value = mock_config
            args = argparse.Namespace(url="http://new-url.com", key=None, model=None, provider=None, json=False)
            cli.config_cmd(args)
            mock_config.save.assert_called_once()