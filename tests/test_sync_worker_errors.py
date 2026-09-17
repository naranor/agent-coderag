import argparse
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.api.models import SyncFileError, SyncResult
from code_rag.entry import cli
from code_rag.services.indexing import sync_project
from code_rag.services.sync import run_rebuild, run_sync
from tests.fake_coderag import fake_coderag_class


def _stack_mock():
    storage = MagicMock()
    parser = MagicMock()
    intel = MagicMock()
    return storage, parser, intel


def _cli_args(*, json_mode, path=None, index_all=True):
    return argparse.Namespace(
        db="test.db",
        onnx=None,
        verbose=False,
        json=json_mode,
        path=path,
        all=index_all,
        force=False,
        allow_build_execution=False,
    )


@pytest.mark.asyncio
async def test_sync_project_reports_stderr_and_returns_failures(capsys):
    storage = MagicMock()
    storage.embedding_model_dirty = False
    storage.embedder = MagicMock()
    storage.get_unit = AsyncMock(return_value=None)
    storage.has_embedding = AsyncMock(return_value=False)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()

    async def distill(path):
        if path == "a.py":
            raise Exception("timeout")
        return []

    parser.distill_file = AsyncMock(side_effect=distill)
    intel = MagicMock()
    failures = await sync_project(storage, parser, intel, ["a.py", "b.py"])
    assert failures == [("a.py", "timeout")]
    assert parser.distill_file.call_count == 2
    assert "Worker failed to sync a.py: timeout" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_sync_project_does_not_hang_when_workers_outnumber_paths():
    """Regression: empty()/get() raced and left a worker blocked forever."""
    import asyncio

    storage = MagicMock()
    storage.embedding_model_dirty = False
    storage.embedder = MagicMock()
    storage.get_unit = AsyncMock(return_value=None)
    storage.has_embedding = AsyncMock(return_value=True)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[])
    intel = MagicMock()

    async def slow_sync(*args, **kwargs):
        await asyncio.sleep(0.01)

    paths = [f"f{i}.py" for i in range(3)]
    with patch("code_rag.services.indexing.sync_file", new=slow_sync):
        failures = await asyncio.wait_for(
            sync_project(storage, parser, intel, paths, max_concurrency=8),
            timeout=2.0,
        )
    assert failures == []


@pytest.mark.asyncio
async def test_run_sync_partial_failure_keeps_discovered_count(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    failed = str(tmp_path / "a.py")
    storage, parser, intel = _stack_mock()
    with patch(
        "code_rag.services.sync.sync_project",
        new=AsyncMock(return_value=[(failed, "timeout")]),
    ) as mock_sp:
        result = await run_sync(
            storage,
            parser,
            intel,
            root=tmp_path,
            path=str(tmp_path),
            index_all=False,
            force=False,
        )
    assert result.status == "error"
    assert result.indexed_files == 2
    assert result.errors == [SyncFileError(file=failed, message="timeout")]
    mock_sp.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sync_single_file_failure_does_not_raise(tmp_path):
    target = tmp_path / "solo.py"
    target.write_text("x = 1\n")
    storage, parser, intel = _stack_mock()
    with patch(
        "code_rag.services.sync.sync_project",
        new=AsyncMock(return_value=[(str(target), "boom")]),
    ):
        result = await run_sync(
            storage,
            parser,
            intel,
            root=tmp_path,
            path=str(target),
            index_all=False,
            force=False,
        )
    assert result.status == "error"
    assert result.indexed_files == 1
    assert result.errors[0].file == str(target)
    assert result.errors[0].message == "boom"


@pytest.mark.asyncio
async def test_run_rebuild_uses_same_error_status(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    storage, parser, intel = _stack_mock()
    with patch(
        "code_rag.services.sync.sync_project",
        new=AsyncMock(return_value=[(str(tmp_path / "a.py"), "embed fail")]),
    ) as mock_sp:
        result = await run_rebuild(storage, parser, intel, root=tmp_path)
    assert result.status == "error"
    assert result.indexed_files == 1
    mock_sp.assert_awaited_once()
    assert mock_sp.await_args.kwargs["index_all"] is True
    assert mock_sp.await_args.kwargs["force_distill"] is True


@pytest.mark.asyncio
async def test_run_sync_all_with_no_files_still_calls_sync_project(tmp_path):
    storage, parser, intel = _stack_mock()
    with patch(
        "code_rag.services.sync.sync_project", new=AsyncMock(return_value=[])
    ) as mock_sp:
        result = await run_sync(
            storage,
            parser,
            intel,
            root=tmp_path,
            path=None,
            index_all=True,
            force=False,
        )
    mock_sp.assert_awaited_once()
    assert mock_sp.await_args.args[3] == []
    assert mock_sp.await_args.kwargs["index_all"] is True
    assert result.status == "success"
    assert result.indexed_files == 0


@pytest.mark.asyncio
async def test_run_sync_empty_dir_without_all_skips_sync_project(tmp_path):
    storage, parser, intel = _stack_mock()
    with patch("code_rag.services.sync.sync_project", new=AsyncMock()) as mock_sp:
        result = await run_sync(
            storage,
            parser,
            intel,
            root=tmp_path,
            path=str(tmp_path),
            index_all=False,
            force=False,
        )
    mock_sp.assert_not_called()
    assert result.status == "success"
    assert result.indexed_files == 0


@pytest.mark.asyncio
async def test_sync_cmd_json_partial_failure_document(capsys):
    result = SyncResult(
        status="error",
        indexed_files=2,
        errors=[
            SyncFileError(
                file="src/a.py", message="Failed to generate embeddings: timeout"
            ),
            SyncFileError(file="src/b.py", message="Embedding count mismatch"),
        ],
    )
    fake_cls, _ = fake_coderag_class(sync=AsyncMock(return_value=result))
    with patch("code_rag.entry.cli.CodeRAG", fake_cls):
        with pytest.raises(SystemExit) as exc:
            await cli.sync_cmd(_cli_args(json_mode=True))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "error",
        "message": "2 file(s) failed",
        "errors": [
            {"file": "src/a.py", "message": "Failed to generate embeddings: timeout"},
            {"file": "src/b.py", "message": "Embedding count mismatch"},
        ],
    }
    assert captured.err == ""


@pytest.mark.asyncio
async def test_sync_cmd_human_partial_failure_stderr(capsys):
    result = SyncResult(
        status="error",
        indexed_files=1,
        errors=[SyncFileError(file="a.py", message="timeout")],
    )
    fake_cls, _ = fake_coderag_class(sync=AsyncMock(return_value=result))
    with patch("code_rag.entry.cli.CodeRAG", fake_cls):
        with pytest.raises(SystemExit) as exc:
            await cli.sync_cmd(_cli_args(json_mode=False))
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "Sync completed with errors."


@pytest.mark.asyncio
async def test_rebuild_cmd_json_partial_failure(capsys):
    result = SyncResult(
        status="error",
        indexed_files=1,
        errors=[SyncFileError(file="a.py", message="boom")],
    )
    fake_cls, _ = fake_coderag_class(rebuild=AsyncMock(return_value=result))
    with patch("code_rag.entry.cli.CodeRAG", fake_cls):
        args = argparse.Namespace(
            db="test.db",
            onnx=None,
            verbose=False,
            json=True,
            allow_build_execution=False,
        )
        with pytest.raises(SystemExit) as exc:
            await cli.rebuild_cmd(args)
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "error",
        "message": "1 file(s) failed",
        "errors": [{"file": "a.py", "message": "boom"}],
    }


@pytest.mark.asyncio
async def test_rebuild_cmd_human_partial_failure(capsys):
    result = SyncResult(
        status="error",
        indexed_files=1,
        errors=[SyncFileError(file="a.py", message="boom")],
    )
    fake_cls, _ = fake_coderag_class(rebuild=AsyncMock(return_value=result))
    with patch("code_rag.entry.cli.CodeRAG", fake_cls):
        args = argparse.Namespace(
            db="test.db",
            onnx=None,
            verbose=False,
            json=False,
            allow_build_execution=False,
        )
        with pytest.raises(SystemExit) as exc:
            await cli.rebuild_cmd(args)
    assert exc.value.code == 1
    assert capsys.readouterr().err.strip() == "Rebuild completed with errors."
