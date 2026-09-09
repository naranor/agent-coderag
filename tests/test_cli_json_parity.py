import argparse
import json
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.entry import cli


@pytest.mark.asyncio
async def test_sync_json_success_shape(tmp_path):
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
            verbose=False,
            json=True,
            path=str(tmp_path),
            all=False,
            force=False,
            allow_build_execution=False,
        )
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            await cli.sync_cmd(args)
            data = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = old
    assert data == {"status": "success", "indexed_files": "auto"}


@pytest.mark.asyncio
async def test_api_json_omits_language():
    mock_manager = MagicMock()
    mock_manager.discovery.extract_api = AsyncMock(return_value="REPORT")
    mock_manager.close = AsyncMock()
    with patch("code_rag.entry.cli.get_manager", return_value=mock_manager):
        args = argparse.Namespace(
            db="test.db",
            onnx=None,
            verbose=False,
            json=True,
            library="pydantic",
            lang="python",
        )
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            await cli.api_cmd(args)
            data = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = old
    assert data == {"library": "pydantic", "report": "REPORT"}
    assert "language" not in data


@pytest.mark.asyncio
async def test_setup_json_success_is_silent(tmp_path):
    with patch("code_rag.entry.cli.get_global_dir", return_value=tmp_path), patch(
        "code_rag.entry.cli.requests.get"
    ) as mock_get:
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        args = argparse.Namespace(force=False, verbose=False, json=True)
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            await cli.setup_cmd(args)
            assert sys.stdout.getvalue() == ""
        finally:
            sys.stdout = old


def test_config_json_update_shape():
    with patch("code_rag.entry.cli.DistillerConfig.load") as mock_load:
        cfg = MagicMock()
        cfg.model_dump.return_value = {"model": "x"}
        mock_load.return_value = cfg
        args = argparse.Namespace(
            url=None, key=None, model="m", provider=None, json=True
        )
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli.config_cmd(args)
            data = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = old
    assert data == {"status": "success", "message": "Config updated"}
