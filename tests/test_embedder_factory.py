from unittest.mock import MagicMock, patch

import pytest

from code_rag.core.exceptions import IntelligenceError
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.intelligence.embedder import LocalOnnxEmbedder
from code_rag.intelligence.factory import create_embedder
from code_rag.intelligence.openai_embedder import OpenAICompatEmbedder


@pytest.mark.asyncio
async def test_xor_base_only_raises():
    cfg = DistillerConfig(embedding_base="http://e", embedding_model=None)
    with pytest.raises(IntelligenceError, match="both be set or both unset"):
        await create_embedder(cfg)


@pytest.mark.asyncio
async def test_xor_model_only_raises():
    cfg = DistillerConfig(embedding_base=None, embedding_model="m")
    with pytest.raises(IntelligenceError, match="both be set or both unset"):
        await create_embedder(cfg)


@pytest.mark.asyncio
async def test_both_set_returns_remote():
    cfg = DistillerConfig(
        embedding_base="http://e",
        embedding_model="emb",
        embedding_key="k",
        embedding_provider="openai",
    )
    embedder = await create_embedder(cfg, onnx_path="ignored.onnx")
    assert isinstance(embedder, OpenAICompatEmbedder)
    assert embedder.model_id == "emb"


@pytest.mark.asyncio
async def test_onnx_plus_remote_warns_stderr(capsys):
    cfg = DistillerConfig(embedding_base="http://e", embedding_model="emb")
    embedder = await create_embedder(cfg, onnx_path="C:\\models\\model.onnx")
    assert isinstance(embedder, OpenAICompatEmbedder)
    captured = capsys.readouterr()
    assert (
        "Warning: --onnx is ignored because remote embeddings are configured."
        in captured.err
    )
    assert captured.out == ""


@pytest.mark.asyncio
async def test_neither_returns_local():
    cfg = DistillerConfig()
    with patch(
        "code_rag.intelligence.factory.LocalOnnxEmbedder",
        return_value=MagicMock(spec=LocalOnnxEmbedder),
    ) as mock_local:
        out = await create_embedder(cfg, onnx_path="custom.onnx")
    mock_local.assert_called_once_with(model_path="custom.onnx")
    assert out is mock_local.return_value
