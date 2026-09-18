import argparse
import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from code_rag.core.exceptions import IntelligenceError
from code_rag.entry import cli
from code_rag.intelligence.distiller import DistillerConfig
from tests.fake_coderag import fake_coderag_class


DISTILLER_KEYS = ("model", "api_base", "api_key", "provider", "temperature")
EMBEDDING_KEYS = (
    "embedding_base",
    "embedding_key",
    "embedding_model",
    "embedding_provider",
)


def test_human_show_two_blocks():
    cfg = DistillerConfig(embedding_base="http://e", embedding_model="m")
    with patch("code_rag.entry.cli.DistillerConfig.load", return_value=cfg):
        args = argparse.Namespace(
            url=None, key=None, model=None, provider=None, json=False
        )
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli.config_cmd(args)
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = old
    assert "Distiller:" in out
    assert "Embedding:" in out
    for key in DISTILLER_KEYS:
        assert key in out
    for key in EMBEDDING_KEYS:
        assert key in out


def test_json_show_is_flat_additive():
    cfg = DistillerConfig(embedding_base="http://e", embedding_model="m")
    with patch("code_rag.entry.cli.DistillerConfig.load", return_value=cfg):
        args = argparse.Namespace(
            url=None, key=None, model=None, provider=None, json=True
        )
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli.config_cmd(args)
            data = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = old
    assert "embedding" not in data or not isinstance(data.get("embedding"), dict)
    assert data["embedding_base"] == "http://e"
    assert data["model"] is None


def test_embedding_flags_write_without_distiller_flags():
    cfg = DistillerConfig()
    with (
        patch.object(DistillerConfig, "save"),
        patch("code_rag.entry.cli.DistillerConfig.load", return_value=cfg),
    ):
        args = argparse.Namespace(
            url=None,
            key=None,
            model=None,
            provider=None,
            json=True,
            embedding_url="http://e",
            embedding_key="k",
            embedding_model="emb",
            embedding_provider="openai",
            clear_embedding=False,
        )
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli.config_cmd(args)
            data = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = old
    assert data == {"status": "success", "message": "Config updated"}
    assert cfg.embedding_base == "http://e"
    assert cfg.embedding_model == "emb"


def test_clear_embedding_flag():
    cfg = DistillerConfig(embedding_base="http://e", embedding_model="m")
    with (
        patch.object(DistillerConfig, "save"),
        patch("code_rag.entry.cli.DistillerConfig.load", return_value=cfg),
    ):
        args = argparse.Namespace(
            url=None,
            key=None,
            model=None,
            provider=None,
            json=True,
            clear_embedding=True,
        )
        cli.config_cmd(args)
    assert cfg.embedding_base is None
    assert cfg.embedding_model is None


def test_partial_embedding_does_not_save():
    cfg = DistillerConfig()
    with (
        patch.object(DistillerConfig, "save") as mock_save,
        patch("code_rag.entry.cli.DistillerConfig.load", return_value=cfg),
    ):
        args = argparse.Namespace(
            url=None,
            key=None,
            model=None,
            provider=None,
            json=False,
            embedding_url="http://only",
            clear_embedding=False,
        )
        with pytest.raises(IntelligenceError, match="both be set or both unset"):
            cli.config_cmd(args)
    mock_save.assert_not_called()


def test_rebuild_cmd_is_not_sync_cmd_alias():
    assert cli.rebuild_cmd is not cli.sync_cmd
    assert cli.rebuild_cmd.__code__ is not cli.sync_cmd.__code__


@pytest.mark.asyncio
async def test_rebuild_cmd_calls_facade_rebuild():
    fake_cls, instances = fake_coderag_class()
    with patch("code_rag.entry.cli.CodeRAG", fake_cls):
        args = argparse.Namespace(
            db="test.db",
            onnx=None,
            verbose=False,
            json=True,
            allow_build_execution=False,
        )
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            await cli.rebuild_cmd(args)
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = old
    instances[0].rebuild.assert_awaited_once()
    assert payload == {"status": "success", "indexed_files": "auto"}
    instances[0].close.assert_awaited()


def test_clear_then_set_on_same_invocation():
    cfg = DistillerConfig(
        embedding_base="http://old",
        embedding_model="old",
        embedding_key="oldk",
        embedding_provider="openai",
    )
    with (
        patch.object(DistillerConfig, "save"),
        patch("code_rag.entry.cli.DistillerConfig.load", return_value=cfg),
    ):
        args = argparse.Namespace(
            url=None,
            key=None,
            model=None,
            provider=None,
            json=True,
            clear_embedding=True,
            embedding_url="http://new",
            embedding_model="new",
            embedding_key=None,
            embedding_provider=None,
        )
        cli.config_cmd(args)
    assert cfg.embedding_base == "http://new"
    assert cfg.embedding_model == "new"
    assert cfg.embedding_key is None


def test_argparse_embedding_dests():
    captured = {}

    def fake_config(args):
        captured["clear"] = args.clear_embedding
        captured["eurl"] = args.embedding_url

    with (
        patch("code_rag.entry.cli.config_cmd", side_effect=fake_config),
        patch(
            "sys.argv",
            [
                "agent-coderag",
                "config",
                "--embedding-url",
                "http://e",
                "--clear-embedding",
            ],
        ),
    ):
        cli.main()
    assert captured["clear"] is True
    assert captured["eurl"] == "http://e"
