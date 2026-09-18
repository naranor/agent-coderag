import pytest
import sys
from io import StringIO
from unittest.mock import MagicMock, patch, AsyncMock
import argparse

from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.entry import cli
from tests.fake_coderag import fake_coderag_class


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
        """Test should_index excludes certain paths using spec."""
        from pathlib import Path

        spec = cli.load_ignore_patterns()
        assert cli.should_index(Path("venv/lib.py"), spec) is False
        assert cli.should_index(Path("sub/venv/lib.py"), spec) is False
        assert cli.should_index(Path("__pycache__/test.py"), spec) is False
        assert cli.should_index(Path(".git/config"), spec) is False
        assert cli.should_index(Path("node_modules/pkg/index.js"), spec) is False


class TestCLISync:
    """Tests for sync command."""

    @pytest.mark.asyncio
    async def test_sync_cmd_with_json(self, tmp_path):
        """Test sync command with JSON output."""
        fake_cls, _ = fake_coderag_class()
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db=str(tmp_path / "test.db"),
                onnx=None,
                verbose=False,
                json=True,
                path=None,
                all=True,
                force=False,
            )

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                await cli.sync_cmd(args)
            finally:
                sys.stdout = old_stdout

    @pytest.mark.asyncio
    async def test_sync_cmd_propagates_build_execution_flag(self, tmp_path):
        """Test the --allow-build-execution opt-in reaches CodeRAG."""
        fake_cls, instances = fake_coderag_class()
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db=str(tmp_path / "test.db"),
                onnx=None,
                verbose=False,
                json=False,
                path=None,
                all=True,
                force=False,
                allow_build_execution=True,
            )

            await cli.sync_cmd(args)

            assert instances[0].init_kwargs["allow_build_execution"] is True

    @pytest.mark.asyncio
    async def test_sync_cmd_file(self, tmp_path):
        """Test sync command for a single file."""
        fake_cls, instances = fake_coderag_class()
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db=str(tmp_path / "test.db"),
                onnx=None,
                verbose=False,
                json=False,
                path=str(test_file),
                all=False,
                force=False,
            )

            await cli.sync_cmd(args)

            instances[0].sync.assert_awaited_once()
            assert instances[0].sync.await_args.kwargs["path"] == str(
                test_file.resolve()
            )


class TestCLISearch:
    """Tests for search command."""

    @pytest.mark.asyncio
    async def test_search_cmd_with_results(self):
        """Test search with results."""
        fake_cls, _ = fake_coderag_class(
            search=AsyncMock(
                return_value=[
                    KnowledgeUnit(
                        id="test.py:test",
                        kind=UnitKind.FUNCTION,
                        name="test",
                        path="test.py",
                        summary="Test function",
                        code_hash="abc123",
                    )
                ]
            )
        )

        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                query="test",
                limit=5,
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
        fake_cls, _ = fake_coderag_class()
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                query="nonexistent",
                limit=5,
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
        fake_cls, _ = fake_coderag_class(
            search=AsyncMock(
                return_value=[
                    KnowledgeUnit(
                        id="test.py:test",
                        kind=UnitKind.FUNCTION,
                        name="test",
                        path="test.py",
                        summary="Test function",
                        code_hash="abc123",
                    )
                ]
            )
        )

        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db", onnx=None, verbose=False, json=True, query="test", limit=5
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
        fake_cls, _ = fake_coderag_class()
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                library="pydantic",
                json=False,
                lang="python",
                verbose=False,
                db="test.db",
                onnx=None,
            )

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                await cli.api_cmd(args)
            finally:
                sys.stdout = old_stdout

    @pytest.mark.asyncio
    async def test_api_cmd_json(self):
        """Test API command with JSON output."""
        fake_cls, _ = fake_coderag_class()
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                library="requests",
                json=True,
                lang="python",
                verbose=False,
                db="test.db",
                onnx=None,
            )

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
        with patch("code_rag.entry.cli.DistillerConfig") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.model_dump.return_value = {"model": "auto"}
            mock_config_cls.load.return_value = mock_config

            args = argparse.Namespace(
                url="http://new-url.com",
                key=None,
                model=None,
                provider=None,
                json=False,
            )

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                cli.config_cmd(args)
            finally:
                sys.stdout = old_stdout

            mock_config.save.assert_called_once()

    def test_config_cmd_json_output(self):
        """Test config JSON output."""
        with patch("code_rag.entry.cli.DistillerConfig") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.model_dump.return_value = {
                "model": "gpt-4",
                "provider": "openai",
            }
            mock_config_cls.load.return_value = mock_config

            args = argparse.Namespace(
                url=None, key=None, model=None, provider=None, json=True
            )

            old_stdout = sys.stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                cli.config_cmd(args)
            finally:
                sys.stdout = old_stdout
