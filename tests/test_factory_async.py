from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.core.exceptions import StorageError
from code_rag.entry import cli
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.services.factory import create_manager


@pytest.mark.asyncio
async def test_create_manager_opens_storage_with_wipe():
    embedder = MagicMock()
    storage = MagicMock()
    manager = MagicMock()
    with patch(
        "code_rag.services.factory.DistillerConfig.load",
        return_value=DistillerConfig(),
    ), patch(
        "code_rag.services.factory.create_embedder",
        new=AsyncMock(return_value=embedder),
    ) as mock_ce, patch(
        "code_rag.services.factory.DuckDBStorage.open",
        new=AsyncMock(return_value=storage),
    ) as mock_open, patch("code_rag.services.factory.Distiller"), patch(
        "code_rag.services.factory.MultiParser"
    ), patch(
        "code_rag.services.factory.CodeRAGManager",
        return_value=manager,
    ):
        out = await create_manager(
            "test.db", onnx_path="model.onnx", allow_build_execution=True, wipe=True
        )
    assert out is manager
    mock_ce.assert_awaited_once()
    mock_open.assert_awaited_once_with("test.db", embedder, wipe=True)


@pytest.mark.asyncio
async def test_get_manager_delegates_and_does_not_construct_cli_embedder():
    sentinel = MagicMock()
    with patch(
        "code_rag.entry.cli.create_manager", new=AsyncMock(return_value=sentinel)
    ) as mock_cm, patch.object(cli, "Embedder") as mock_embedder:
        out = await cli.get_manager("db.db", onnx_path="x.onnx", wipe=True)
    assert out is sentinel
    mock_cm.assert_awaited_once_with(
        "db.db", "x.onnx", allow_build_execution=False, wipe=True
    )
    mock_embedder.assert_not_called()


@pytest.mark.asyncio
async def test_create_manager_selects_remote_via_same_rule():
    cfg = DistillerConfig(embedding_base="http://e", embedding_model="emb")
    remote = MagicMock(name="remote")
    with patch(
        "code_rag.services.factory.DistillerConfig.load", return_value=cfg
    ), patch(
        "code_rag.services.factory.create_embedder",
        new=AsyncMock(return_value=remote),
    ) as mock_ce, patch(
        "code_rag.services.factory.DuckDBStorage.open",
        new=AsyncMock(return_value=MagicMock()),
    ), patch("code_rag.services.factory.Distiller"), patch(
        "code_rag.services.factory.MultiParser"
    ), patch(
        "code_rag.services.factory.CodeRAGManager",
        return_value=MagicMock(),
    ):
        await create_manager("db.db")
    assert mock_ce.await_args.args[0] is cfg


@pytest.mark.asyncio
async def test_create_manager_closes_embedder_when_open_fails():
    embedder = MagicMock()
    embedder.close = AsyncMock()
    with patch(
        "code_rag.services.factory.DistillerConfig.load",
        return_value=DistillerConfig(),
    ), patch(
        "code_rag.services.factory.create_embedder",
        new=AsyncMock(return_value=embedder),
    ), patch(
        "code_rag.services.factory.DuckDBStorage.open",
        new=AsyncMock(side_effect=StorageError("open failed")),
    ), patch("code_rag.services.factory.Distiller"), patch(
        "code_rag.services.factory.MultiParser"
    ), patch(
        "code_rag.services.factory.CodeRAGManager",
    ):
        with pytest.raises(StorageError, match="open failed"):
            await create_manager("db.db")
    embedder.close.assert_awaited_once()
