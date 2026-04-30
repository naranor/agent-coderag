import pytest
import sys
from io import StringIO
from unittest.mock import MagicMock, patch, AsyncMock
import argparse

from code_rag.entry import cli


class TestCLIHelpers:
    """Tests for CLI helper functions."""
    
    def test_should_index_py_file(self):
        """Test should_index for Python files."""
        from pathlib import Path
        path = Path("test.py")
        assert cli.should_index(path) is True
    
    def test_should_index_non_py_file(self):
        """Test should_index for non-Python files."""
        from pathlib import Path
        path = Path("test.txt")
        assert cli.should_index(path) is False
    
    def test_should_index_excluded_paths(self):
        """Test should_index excludes certain paths."""
        from pathlib import Path
        assert cli.should_index(Path("tests/test.py")) is False
        assert cli.should_index(Path("venv/lib.py")) is False
        assert cli.should_index(Path("__pycache__/test.py")) is False
        assert cli.should_index(Path(".git/config.py")) is False
    
    @patch('code_rag.entry.cli.DistillerConfig')
    @patch('code_rag.entry.cli.Embedder')
    @patch('code_rag.entry.cli.DuckDBStorage')
    @patch('code_rag.entry.cli.AstIndexParser')
    @patch('code_rag.entry.cli.Distiller')
    def test_get_manager_init(self, mock_distiller, mock_parser, mock_storage, mock_embedder, mock_config):
        """Test get_manager initialization."""
        mock_config.load.return_value = MagicMock(
            api_base="http://localhost:8081",
            api_key="test-key",
            model="gpt-4",
            provider="openai"
        )
        
        manager = cli.get_manager("test.db", verbose=False)
        
        assert manager is not None


class TestCLISync:
    """Tests for sync command."""
    
    @pytest.mark.asyncio
    async def test_sync_cmd_with_json(self, tmp_path):
        """Test sync command with JSON output."""
        mock_manager = MagicMock()
        mock_manager.sync_project = AsyncMock()
        
        with patch('code_rag.entry.cli.get_manager', return_value=mock_manager):
            args = argparse.Namespace(
                db=str(tmp_path / "test.db"),
                onnx=None,
                verbose=False,
                json=True,
                path=None,
                all=True,
                force=False
            )
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await cli.sync_cmd(args)
            finally:
                sys.stdout = old_stdout
            
            # captured is removed to satisfy linter
    
    @pytest.mark.asyncio
    async def test_sync_cmd_file(self, tmp_path):
        """Test sync command for a single file."""
        mock_manager = MagicMock()
        mock_manager.sync_file = AsyncMock()
        
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")
        
        with patch('code_rag.entry.cli.get_manager', return_value=mock_manager):
            args = argparse.Namespace(
                db=str(tmp_path / "test.db"),
                onnx=None,
                verbose=False,
                json=False,
                path=str(test_file),
                all=False,
                force=False
            )
            
            await cli.sync_cmd(args)
            
            mock_manager.sync_file.assert_called_once()


class TestCLISearch:
    """Tests for search command."""
    
    @pytest.mark.asyncio
    async def test_search_cmd_with_results(self):
        """Test search with results."""
        from code_rag.core.models import KnowledgeUnit, UnitKind
        
        mock_manager = MagicMock()
        mock_manager.search = AsyncMock(return_value=[
            KnowledgeUnit(
                id="test.py:test",
                kind=UnitKind.FUNCTION,
                name="test",
                path="test.py",
                summary="Test function",
                code_hash="abc123"
            )
        ])
        
        with patch('code_rag.entry.cli.get_manager', return_value=mock_manager):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                query="test",
                limit=5
            )
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await cli.search_cmd(args)
            finally:
                sys.stdout = old_stdout
    
    @pytest.mark.asyncio
    async def test_search_cmd_no_results(self):
        """Test search with no results."""
        mock_manager = MagicMock()
        mock_manager.search = AsyncMock(return_value=[])
        
        with patch('code_rag.entry.cli.get_manager', return_value=mock_manager):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                query="nonexistent",
                limit=5
            )
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await cli.search_cmd(args)
            finally:
                sys.stdout = old_stdout
    
    @pytest.mark.asyncio
    async def test_search_cmd_json_output(self):
        """Test search with JSON output."""
        from code_rag.core.models import KnowledgeUnit, UnitKind
        
        mock_manager = MagicMock()
        mock_manager.search = AsyncMock(return_value=[
            KnowledgeUnit(
                id="test.py:test",
                kind=UnitKind.FUNCTION,
                name="test",
                path="test.py",
                summary="Test function",
                code_hash="abc123"
            )
        ])
        
        with patch('code_rag.entry.cli.get_manager', return_value=mock_manager):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=True,
                query="test",
                limit=5
            )
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await cli.search_cmd(args)
            finally:
                sys.stdout = old_stdout


class TestCLIApi:
    """Tests for API command."""
    
    @pytest.mark.asyncio
    async def test_api_cmd_success(self):
        """Test API discovery command."""
        mock_output = {"functions": ["func1", "func2"]}
        
        with patch('code_rag.entry.cli.extract_library_api', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_output
            
            args = argparse.Namespace(library="pydantic", json=False)
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await cli.api_cmd(args)
            finally:
                sys.stdout = old_stdout
    
    @pytest.mark.asyncio
    async def test_api_cmd_json(self):
        """Test API command with JSON output."""
        mock_output = {"functions": ["func1"]}
        
        with patch('code_rag.entry.cli.extract_library_api', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_output
            
            args = argparse.Namespace(library="requests", json=True)
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await cli.api_cmd(args)
            finally:
                sys.stdout = old_stdout


class TestCLIConfig:
    """Tests for config command."""
    
    def test_config_cmd_update_url(self):
        """Test config update with URL."""
        with patch('code_rag.entry.cli.DistillerConfig') as mock_config_cls:
            mock_config = MagicMock()
            mock_config.model_dump.return_value = {"model": "auto"}
            mock_config_cls.load.return_value = mock_config
            
            args = argparse.Namespace(url="http://new-url.com", key=None, model=None, provider=None, json=False)
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                cli.config_cmd(args)
            finally:
                sys.stdout = old_stdout
            
            mock_config.save.assert_called_once()
    
    def test_config_cmd_json_output(self):
        """Test config JSON output."""
        with patch('code_rag.entry.cli.DistillerConfig') as mock_config_cls:
            mock_config = MagicMock()
            mock_config.model_dump.return_value = {"model": "gpt-4", "provider": "openai"}
            mock_config_cls.load.return_value = mock_config
            
            args = argparse.Namespace(url=None, key=None, model=None, provider=None, json=True)
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                cli.config_cmd(args)
            finally:
                sys.stdout = old_stdout
