import pytest
import sys
import argparse
import json
from io import StringIO
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from code_rag.entry import args as cli_args
from code_rag.entry import cli
from code_rag.core.exceptions import CodeRAGError
from code_rag.api.models import ApiReport
from tests.fake_coderag import fake_coderag_class


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
        fake_cls, _ = fake_coderag_class()
        with patch("code_rag.entry.cli.CodeRAG", fake_cls), patch(
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
        fake_cls, instances = fake_coderag_class(
            sync=AsyncMock(side_effect=Exception("Critical Failure"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
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
                with pytest.raises(SystemExit) as exc:
                    await cli.sync_cmd(args)
                assert exc.value.code == 1
                data = json.loads(sys.stdout.getvalue())
                assert data["status"] == "error"
                assert "Critical Failure" in data["message"]
            finally:
                sys.stdout = old_stdout
            instances[0].close.assert_awaited()

    @pytest.mark.asyncio
    async def test_search_cmd_json_error(self):
        fake_cls, instances = fake_coderag_class(
            search=AsyncMock(side_effect=Exception("Database file not found: x.db"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="x.db",
                onnx=None,
                verbose=False,
                json=True,
                query="auth",
                limit=5,
            )
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    await cli.search_cmd(args)
                assert exc.value.code == 1
                data = json.loads(sys.stdout.getvalue())
                assert data["status"] == "error"
                assert "Database file not found" in data["message"]
                assert "code" not in data
            finally:
                sys.stdout = old_stdout
            instances[0].close.assert_awaited()

    @pytest.mark.asyncio
    async def test_search_cmd_json_error_includes_code(self):
        from code_rag.core.error_codes import ErrorCode
        from code_rag.core.exceptions import StorageError

        fake_cls, instances = fake_coderag_class(
            search=AsyncMock(
                side_effect=StorageError(
                    "Embeddings table is missing; run sync (CodeRAG.sync) before search.",
                    code=ErrorCode.EMBEDDINGS_MISSING,
                )
            )
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="x.db",
                onnx=None,
                verbose=False,
                json=True,
                query="auth",
                limit=5,
            )
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    await cli.search_cmd(args)
                assert exc.value.code == 1
                data = json.loads(sys.stdout.getvalue())
                assert data["status"] == "error"
                assert data["code"] == "EMBEDDINGS_MISSING"
                assert "sync" in data["message"]
            finally:
                sys.stdout = old_stdout
            instances[0].close.assert_awaited()

    @pytest.mark.asyncio
    async def test_search_cmd_human_error(self):
        fake_cls, _ = fake_coderag_class(
            search=AsyncMock(side_effect=Exception("Database file not found: x.db"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="x.db",
                onnx=None,
                verbose=False,
                json=False,
                query="auth",
                limit=5,
            )
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    await cli.search_cmd(args)
                assert exc.value.code == 1
                err = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
        assert "Database file not found" in err

    @pytest.mark.asyncio
    async def test_sync_cmd_human_error(self):
        fake_cls, _ = fake_coderag_class(
            sync=AsyncMock(side_effect=Exception("Critical Failure"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                path=None,
                all=True,
                force=False,
            )
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    await cli.sync_cmd(args)
                assert exc.value.code == 1
                assert "Critical Failure" in sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr

    @pytest.mark.asyncio
    async def test_rebuild_cmd_json_error(self):
        fake_cls, instances = fake_coderag_class(
            rebuild=AsyncMock(side_effect=Exception("Rebuild boom"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=True,
                allow_build_execution=False,
            )
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    await cli.rebuild_cmd(args)
                assert exc.value.code == 1
                data = json.loads(sys.stdout.getvalue())
                assert data["status"] == "error"
                assert "Rebuild boom" in data["message"]
            finally:
                sys.stdout = old_stdout
            instances[0].close.assert_awaited()

    @pytest.mark.asyncio
    async def test_rebuild_cmd_human_error(self):
        fake_cls, _ = fake_coderag_class(
            rebuild=AsyncMock(side_effect=Exception("Rebuild boom"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                allow_build_execution=False,
            )
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    await cli.rebuild_cmd(args)
                assert exc.value.code == 1
                assert "Rebuild boom" in sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr

    @pytest.mark.asyncio
    async def test_api_cmd_logic(self):
        """Test api command logic."""
        fake_cls, instances = fake_coderag_class(
            api=AsyncMock(
                return_value=ApiReport(
                    library="testlib", language="python", report="API Report"
                )
            )
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                library="testlib",
                lang="python",
            )
            await cli.api_cmd(args)
            instances[0].api.assert_awaited_once_with("testlib", lang="python")

    @pytest.mark.asyncio
    async def test_api_cmd_json_error(self):
        fake_cls, _ = fake_coderag_class(
            api=AsyncMock(side_effect=Exception("api down"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=True,
                library="testlib",
                lang="python",
            )
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                await cli.api_cmd(args)
                data = json.loads(sys.stdout.getvalue())
            finally:
                sys.stdout = old_stdout
        assert data["status"] == "error"
        assert "api down" in data["message"]

    @pytest.mark.asyncio
    async def test_api_cmd_human_error(self):
        fake_cls, _ = fake_coderag_class(
            api=AsyncMock(side_effect=Exception("api down"))
        )
        with patch("code_rag.entry.cli.CodeRAG", fake_cls):
            args = argparse.Namespace(
                db="test.db",
                onnx=None,
                verbose=False,
                json=False,
                library="testlib",
                lang=None,
            )
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                await cli.api_cmd(args)
                err = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
        assert "api down" in err

    def test_cli_main_exception_handling(self):
        """Test main entry point handles exceptions gracefully."""
        with patch(
            "code_rag.entry.args.argparse.ArgumentParser.parse_args"
        ) as mock_parse, patch("code_rag.entry.cli.asyncio.run"):
            mock_parse.side_effect = CodeRAGError("Known Error")

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                with pytest.raises(SystemExit) as exc:
                    cli_args.main()
                assert exc.value.code == 1
                assert "Known Error" in sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr

    def test_should_index_ignores(self):
        ignore_spec = MagicMock()
        ignore_spec.match_file.return_value = True
        assert cli.should_index(Path("ignored.py"), ignore_spec) is False
