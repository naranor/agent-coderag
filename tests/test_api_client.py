import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from code_rag.api.client import CodeRAG
from code_rag.api.models import ApiReport, SetupResult, SyncResult
from code_rag.services.config import load_or_update_config
from code_rag.services.discovery_api import run_api
from code_rag.services.setup import run_setup
from code_rag.services.sync import run_sync


def test_public_imports():
    from code_rag import (
        CodeRAG as PublicCodeRAG,
        UnitKind,
    )

    assert PublicCodeRAG is not None
    assert UnitKind is not None


def test_public_all_does_not_leak_internals():
    import code_rag

    leaked = {"DuckDBStorage", "MultiParser", "Embedder", "Distiller"}
    assert leaked.isdisjoint(set(code_rag.__all__))


def test_config_read_returns_loaded_model():
    with patch("code_rag.services.config.DistillerConfig.load") as mock_load:
        cfg = MagicMock(name="cfg")
        mock_load.return_value = cfg
        assert load_or_update_config() is cfg
        cfg.save.assert_not_called()


def test_config_update_maps_url_key_and_saves():
    with patch("code_rag.services.config.DistillerConfig.load") as mock_load:
        cfg = MagicMock()
        mock_load.return_value = cfg
        out = load_or_update_config(
            url="http://x", key="k", model="m", provider="ollama"
        )
        assert out is cfg
        assert cfg.api_base == "http://x"
        assert cfg.api_key == "k"
        assert cfg.model == "m"
        assert cfg.provider == "ollama"
        cfg.save.assert_called_once()


@pytest.mark.asyncio
async def test_setup_downloads_missing_files(tmp_path):
    with patch("code_rag.services.setup.get_global_dir", return_value=tmp_path), patch(
        "code_rag.services.setup.requests.get"
    ) as mock_get:
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = await run_setup(force=False)
        assert isinstance(result, SetupResult)
        assert (tmp_path / "models" / "mini-lm" / "model.onnx").exists()
        assert "model.onnx" in result.downloaded
        assert "tokenizer.json" in result.downloaded


@pytest.mark.asyncio
async def test_setup_skips_existing_files(tmp_path):
    model_dir = tmp_path / "models" / "mini-lm"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"x")
    (model_dir / "tokenizer.json").write_bytes(b"y")

    with patch("code_rag.services.setup.get_global_dir", return_value=tmp_path), patch(
        "code_rag.services.setup.requests.get"
    ) as mock_get:
        result = await run_setup(force=False)
        mock_get.assert_not_called()
        assert result.downloaded == []
        assert set(result.skipped) == {"model.onnx", "tokenizer.json"}


@pytest.mark.asyncio
async def test_run_api_defaults_language_python():
    discovery = MagicMock()
    discovery.extract_api = AsyncMock(return_value="REPORT")
    out = await run_api(discovery, "pydantic", lang=None)
    assert isinstance(out, ApiReport)
    assert out.language == "python"
    assert out.library == "pydantic"
    assert out.report == "REPORT"
    discovery.extract_api.assert_awaited_once_with("pydantic", language="python")


@pytest.mark.asyncio
async def test_run_sync_noop_when_no_path_and_not_index_all():
    manager = MagicMock()
    manager.sync_dependencies = AsyncMock()
    manager.sync_project = AsyncMock()
    manager.sync_file = AsyncMock()
    result = await run_sync(
        manager, root=Path("."), path=None, index_all=False, force=False
    )
    assert isinstance(result, SyncResult)
    assert result.status == "success"
    assert result.indexed_files == 0
    manager.sync_project.assert_not_called()
    manager.sync_file.assert_not_called()


@pytest.mark.asyncio
async def test_config_does_not_create_stack():
    with patch("code_rag.api.client.create_stack", new=AsyncMock()) as mock_stack:
        rag = CodeRAG()
        with patch(
            "code_rag.api.client.load_or_update_config", return_value=MagicMock()
        ):
            await rag.config()
        mock_stack.assert_not_called()


@pytest.mark.asyncio
async def test_setup_does_not_create_stack():
    with patch("code_rag.api.client.create_stack", new=AsyncMock()) as mock_stack:
        rag = CodeRAG()
        with patch(
            "code_rag.api.client.run_setup",
            new=AsyncMock(return_value=MagicMock()),
        ):
            await rag.setup()
        mock_stack.assert_not_called()


@pytest.mark.asyncio
async def test_sync_noop_without_path_and_index_all_false():
    rag = CodeRAG()
    with patch("code_rag.api.client.create_stack", new=AsyncMock()) as mock_stack:
        result = await rag.sync(path=None, index_all=False, force=False)
        assert result.indexed_files == 0
        mock_stack.assert_not_called()


@pytest.mark.asyncio
async def test_search_closes_storage_then_embedder():
    embedder = MagicMock()
    embedder.close = AsyncMock()
    parser = MagicMock()
    distiller = MagicMock()
    storage = MagicMock()
    storage.close = AsyncMock()
    with patch(
        "code_rag.api.client.create_stack",
        new=AsyncMock(return_value=(embedder, parser, distiller)),
    ), patch(
        "code_rag.api.client.open_db_connection",
        new=AsyncMock(return_value=storage),
    ), patch("code_rag.api.client.build_manager", return_value=MagicMock()), patch(
        "code_rag.api.client.run_search", new=AsyncMock(return_value=[])
    ) as mock_search:
        async with CodeRAG() as rag:
            await rag.search("q", limit=3)
        mock_search.assert_awaited_once()
        assert mock_search.await_args.kwargs["limit"] == 3
        storage.close.assert_awaited()
        embedder.close.assert_awaited()
