import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from code_rag.core.exceptions import CodeRAGError
from code_rag.entry import args as cli_args
from code_rag.entry import cli


def test_help_entry_does_not_import_heavy_stack():
    code = """
import sys
from code_rag.entry.args import main
assert "litellm" not in sys.modules
assert "duckdb" not in sys.modules
assert "onnxruntime" not in sys.modules
sys.argv = ["agent-coderag", "--help"]
main()
assert "litellm" not in sys.modules
assert "duckdb" not in sys.modules
assert "onnxruntime" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_main_help_and_verbose_do_not_dispatch(capsys):
    with patch("sys.argv", ["agent-coderag"]):
        cli_args.main()
    assert "CodeRAG" in capsys.readouterr().out

    with patch("sys.argv", ["agent-coderag", "--verbose"]):
        cli_args.main()
    assert "usage:" in capsys.readouterr().out


def test_main_dispatches_subcommand():
    seen = []

    def dispatch(args):
        seen.append(args.command)

    real_import = cli_args.importlib.import_module

    def fake_import(name, package=None):
        if name == "code_rag.entry.cli":
            return SimpleNamespace(dispatch=dispatch)
        return real_import(name, package)

    with (
        patch("sys.argv", ["agent-coderag", "rebuild"]),
        patch.object(cli_args.importlib, "import_module", side_effect=fake_import),
    ):
        cli_args.main()
    assert seen == ["rebuild"]


def test_main_keyboard_interrupt_exits_zero():
    with patch.object(
        cli_args.argparse.ArgumentParser, "parse_args", side_effect=KeyboardInterrupt
    ):
        with pytest.raises(SystemExit) as exc:
            cli_args.main()
    assert exc.value.code == 0


def test_main_unexpected_error_exits_one():
    with patch.object(
        cli_args.argparse.ArgumentParser,
        "parse_args",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(SystemExit) as exc:
            cli_args.main()
    assert exc.value.code == 1


def test_dispatch_routes_every_command():
    async def _noop():
        return None

    def fake_cmd(_args):
        return _noop()

    def close_run(coro):
        coro.close()

    with (
        patch("code_rag.entry.cli.sync_cmd", side_effect=fake_cmd),
        patch("code_rag.entry.cli.search_cmd", side_effect=fake_cmd),
        patch("code_rag.entry.cli.api_cmd", side_effect=fake_cmd),
        patch("code_rag.entry.cli.setup_cmd", side_effect=fake_cmd),
        patch("code_rag.entry.cli.rebuild_cmd", side_effect=fake_cmd),
        patch("code_rag.entry.cli.config_cmd") as config_cmd,
        patch("code_rag.entry.cli.asyncio.run", side_effect=close_run) as run,
    ):
        for command in ("sync", "search", "api", "setup", "rebuild"):
            cli.dispatch(SimpleNamespace(command=command))
        cli.dispatch(SimpleNamespace(command="config"))
    assert run.call_count == 5
    config_cmd.assert_called_once()


def test_emit_json_error_code_without_value(capsys):
    err = CodeRAGError("missing", code=object())
    cli._emit_json_error(err)
    assert '"code":' in capsys.readouterr().out


def test_package_getattr_unknown_name():
    import code_rag

    with pytest.raises(AttributeError, match="no attribute"):
        getattr(code_rag, "NotAnExport")
