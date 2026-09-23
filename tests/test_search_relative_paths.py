from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.api.client import CodeRAG
from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.entry.args import build_parser
from code_rag.entry.cli import search_cmd
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.services.search import format_unit_path
from tests.embedder_stubs import StubEmbedder
from tests.fake_coderag import fake_coderag_class


def _unit(path: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=f"{path}:alpha",
        name="alpha",
        kind=UnitKind.FUNCTION,
        path=path,
        code_hash="h",
    )


def test_format_unit_path_table(tmp_path: Path):
    absolute = str((tmp_path / "old" / "a.py").resolve())
    assert format_unit_path(absolute, root=tmp_path, relative_paths=True) == absolute
    assert format_unit_path(absolute, root=tmp_path, relative_paths=False) == absolute
    assert (
        format_unit_path("src/a.py", root=tmp_path, relative_paths=True) == "src/a.py"
    )
    assert format_unit_path("src/a.py", root=tmp_path, relative_paths=False) == str(
        (tmp_path / "src" / "a.py").resolve()
    )


def test_config_field_defaults_false_and_roundtrips():
    assert DistillerConfig().relative_paths is False
    loaded = DistillerConfig(**DistillerConfig(relative_paths=True).model_dump())
    assert loaded.relative_paths is True
    assert DistillerConfig.model_validate({}).relative_paths is False


def test_constructor_precedence(tmp_path: Path):
    configured = DistillerConfig(relative_paths=True)
    with patch(
        "code_rag.api.client.DistillerConfig.load", return_value=configured
    ) as load:
        forced_off = CodeRAG(db=str(tmp_path / "a.db"), relative_paths=False)
        assert forced_off._relative_paths is False
        load.assert_not_called()
        from_config = CodeRAG(db=str(tmp_path / "b.db"))
        assert from_config._relative_paths is True
        load.assert_called()


@pytest.mark.asyncio
async def test_search_formats_stored_paths(tmp_path: Path, monkeypatch):
    """Search formats stored paths. The DB file is not created; open is patched.

    Same pattern as ``tests/test_api_rebuild_wipe.py::test_search_uses_read_only_connection``.
    ``CodeRAG.search`` calls ``create_stack`` and ``open_db_connection`` before ``run_search``.
    A missing db file would raise ``StorageError`` before formatting.
    """
    storage = MagicMock()
    storage.close = AsyncMock()

    async def tracking_open(
        path, emb, *, mode, connect_timeout_seconds=5.0, wipe=False
    ):
        return storage

    monkeypatch.setattr(
        "code_rag.api.client.create_stack",
        AsyncMock(return_value=(StubEmbedder(dim=8), MagicMock(), MagicMock())),
    )
    monkeypatch.setattr("code_rag.api.client.open_db_connection", tracking_open)
    absolute = str((tmp_path / "gone.py").resolve())
    monkeypatch.setattr(
        "code_rag.api.client.run_search",
        AsyncMock(return_value=[_unit(absolute), _unit("src/a.py")]),
    )
    rag = CodeRAG(db=str(tmp_path / "a.db"), root=tmp_path, relative_paths=False)
    hits = await rag.search("q")
    assert hits[0].path == absolute
    assert hits[1].path == str((tmp_path / "src" / "a.py").resolve())

    monkeypatch.setattr(
        "code_rag.api.client.run_search",
        AsyncMock(return_value=[_unit(absolute), _unit("src/a.py")]),
    )
    rag_rel = CodeRAG(db=str(tmp_path / "b.db"), root=tmp_path, relative_paths=True)
    hits = await rag_rel.search("q")
    assert hits[0].path == absolute
    assert hits[1].path == "src/a.py"


@pytest.mark.asyncio
async def test_cli_relative_paths_flag():
    flagged = build_parser().parse_args(["search", "auth", "--relative-paths"])
    omitted = build_parser().parse_args(["search", "auth"])
    assert flagged.relative_paths is True
    assert omitted.relative_paths is None

    fake, instances = fake_coderag_class()
    with patch("code_rag.entry.cli.CodeRAG", fake):
        await search_cmd(flagged)
        await search_cmd(omitted)
    assert instances[0].init_kwargs["relative_paths"] is True
    assert instances[1].init_kwargs["relative_paths"] is None
