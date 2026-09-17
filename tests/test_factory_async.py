from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.intelligence.distiller import DistillerConfig
from code_rag.services.factory import create_stack


@pytest.mark.asyncio
async def test_create_stack_returns_embedder_parser_distiller():
    embedder = MagicMock()
    parser = MagicMock()
    distiller = MagicMock()
    cfg = DistillerConfig()
    with patch(
        "code_rag.services.factory.DistillerConfig.load",
        return_value=cfg,
    ), patch(
        "code_rag.services.factory.create_embedder",
        new=AsyncMock(return_value=embedder),
    ) as mock_ce, patch(
        "code_rag.services.factory.Distiller",
        return_value=distiller,
    ) as mock_distiller, patch(
        "code_rag.services.factory.MultiParser",
        return_value=parser,
    ) as mock_parser:
        out_embedder, out_parser, out_distiller = await create_stack(
            onnx_path="model.onnx"
        )

    assert out_embedder is embedder
    assert out_parser is parser
    assert out_distiller is distiller
    mock_ce.assert_awaited_once_with(cfg, onnx_path="model.onnx")
    mock_distiller.assert_called_once_with(cfg)
    mock_parser.assert_called_once()


@pytest.mark.asyncio
async def test_create_stack_passes_config_to_embedder():
    cfg = DistillerConfig(embedding_base="http://e", embedding_model="emb")
    remote = MagicMock(name="remote")
    with patch(
        "code_rag.services.factory.DistillerConfig.load", return_value=cfg
    ), patch(
        "code_rag.services.factory.create_embedder",
        new=AsyncMock(return_value=remote),
    ) as mock_ce, patch("code_rag.services.factory.Distiller"), patch(
        "code_rag.services.factory.MultiParser"
    ):
        await create_stack()
    assert mock_ce.await_args.args[0] is cfg
