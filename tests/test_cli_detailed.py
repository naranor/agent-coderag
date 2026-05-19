import pytest
import sys
import argparse
import json
from io import StringIO
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from code_rag.entry import cli
from code_rag.core.exceptions import CodeRAGError


class TestCLIDetailed:
    """Detailed tests for CLI commands to increase coverage."""

    @pytest.mark.asyncio
    async def test_setup_cmd_success(self, tmp_path):
        """Test setup command correctly downloads models."""
        with patch("code_rag.entry.cli.get_global_dir", return_value=tmp_path), patch(
            "code_rag.entry.cli.requests.get"
        ) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.iter_content.return_value = [b"data"]
            mock_get.return_value = mock_resp

            # Added json=False to match cli implementation
            args = argparse.Namespace(force=False, verbose=False, json=False)
            await cli.setup_cmd(args)

            assert (tmp_path / "models" / "mini-lm" / "model.onnx").exists()
            assert (tmp_path / "models" / "mini-lm" / "tokenizer.json").exists()

    @pytest.mark.asyncio
    async def test_setup_cmd_already_exists(self, tmp_path):
        """Test setup command skips existing files."""
        model_dir = tmp_path / "models" / "mini-lm"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").touch()
        (model_dir / "tokenizer.json").touch()

        with patch("code_rag.entry.cli.get_global_dir", return_value=tmp_path), patch(
            "code_rag.entry.cli.requests.get"
        ) as mock_get:
            args = argparse.Namespace(force=False, verbose=False, json=False)
            await cli.setup_cmd(args)
            mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_cmd_download_error(self, tmp_path):
        """Test setup command handles download errors."""
        with patch("code_rag.entry.cli.get_global_dir", return_value=tmp_path), patch(
            "code_rag.entry.cli.requests.get"
        ) as mock_get:
            mock_get.side_effect = Exception("Network Down")
            args = argparse.Namespace(force=False, verbose=False, json=False)
            await cli.setup_cmd(args)
            # Should not crash, just print error

    def test_config_cmd_no_args_shows_current(self):
        """Test config command with no args shows current state."""
        with patch("code_rag.entry.cli.DistillerConfig.load") as mock_load:
            mock_cfg = MagicMock()
            mock_cfg.model_dump.return_value = {"model": "test-model"}
            mock_load.return_value = mock_cfg

            args = argparse.Namespace(
                url=None, key=None, model=None, provider=None, json=False
            )

            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                cli.config_cmd(args)
                output = sys.stdout.getvalue()
                assert "test-model" in output
                assert "Config updated" not in output
            finally:
                sys.stdout = old_stdout

    def test_config_cmd_json(self):
        """Test config command output in JSON."""
        with patch("code_rag.entry.cli.DistillerConfig.load") as mock_load:
            mock_cfg = MagicMock()
            mock_cfg.model_dump.return_value = {"model": "json-model"}
            mock_load.return_value = mock_cfg

            args = argparse.Namespace(
                url=None, key=None, model=None, provider=None, json=True
            )

            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                cli.config_cmd(args)
                data = json.loads(sys.stdout.getvalue())
                assert data["model"] == "json-model"
            finally:
                sys.stdout = old_stdout

    @pytest.mark.asyncio
    async def test_sync_cmd_json_success(self, tmp_path):
        mock_manager = MagicMock()
        mock_manager.sync_dependencies = AsyncMock()
        mock_manager.sync_project = AsyncMock()
        mock_manager.close = AsyncMock()

        with patch("code_rag.entry.cli.get_manager", return_value=mock_manager), patch(
            "code_rag.entry.cli.validate_path", return_value=tmp_path
        ):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=True,
                json=True,
                path=str(tmp_path),
                all=False,
                force=False,
            )
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                await cli.sync_cmd(args)
                data = json.loads(sys.stdout.getvalue())
                assert data["status"] == "success"
            finally:
                sys.stdout = old_stdout

    @pytest.mark.asyncio
    async def test_sync_cmd_json_error(self, tmp_path):
        mock_manager = MagicMock()
        mock_manager.sync_dependencies = AsyncMock(
            side_effect=Exception("Critical Failure")
        )
        mock_manager.close = AsyncMock()

        with patch("code_rag.entry.cli.get_manager", return_value=mock_manager):
            args = argparse.Namespace(
                db="test.db",
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
                data = json.loads(sys.stdout.getvalue())
                assert data["status"] == "error"
                assert "Critical Failure" in data["message"]
            finally:
                sys.stdout = old_stdout

    @pytest.mark.asyncio
    async def test_api_cmd_logic(self):
        """Test api command logic."""
        mock_manager = MagicMock()
        mock_manager.discovery.extract_api = AsyncMock(return_value="API Report")
        mock_manager.close = AsyncMock()

        with patch("code_rag.entry.cli.get_manager", return_value=mock_manager):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                library="testlib",
                lang="python",
            )
            await cli.api_cmd(args)
            # Correct assertion with keyword argument
            mock_manager.discovery.extract_api.assert_called_with(
                "testlib", language="python"
            )

    def test_cli_main_exception_handling(self):
        """Test main entry point handles exceptions gracefully."""
        with patch(
            "code_rag.entry.cli.argparse.ArgumentParser.parse_args"
        ) as mock_parse, patch("code_rag.entry.cli.asyncio.run"):
            mock_parse.side_effect = CodeRAGError("Known Error")

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    cli.main()
                assert exc.value.code == 1
                assert "Known Error" in sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr

    def test_should_index_ignores(self):
        ignore_spec = MagicMock()
        ignore_spec.match_file.return_value = True
        assert cli.should_index(Path("ignored.py"), ignore_spec) is False
