import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from code_rag.core.constants import EMBEDDING_REQUEST_TIMEOUT
from code_rag.core.exceptions import IntelligenceError
from code_rag.intelligence.openai_embedder import OpenAICompatEmbedder


def _resp(vectors):
    data = []
    for vec in vectors:
        item = MagicMock()
        item.embedding = vec
        data.append(item)
    resp = MagicMock()
    resp.data = data
    return resp


@pytest.mark.asyncio
async def test_unbound_probe_does_not_check_length_and_defines_via_bind():
    embedder = OpenAICompatEmbedder(
        api_base="http://e", model="emb-3", api_key="secret"
    )
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(return_value=_resp([[3.0, 4.0, 0.0]])),
    ) as mock_emb:
        rows = await embedder.aembed(["probe"])
    mock_emb.assert_awaited()
    kwargs = mock_emb.call_args.kwargs
    assert kwargs["model"] == "emb-3"
    assert kwargs["input"] == ["probe"]
    assert kwargs["api_base"] == "http://e"
    assert kwargs["timeout"] == EMBEDDING_REQUEST_TIMEOUT
    assert kwargs["api_key"] == "secret"
    embedder.bind_dimension(len(rows[0]))
    assert embedder.dimension == 3
    np.testing.assert_allclose(np.linalg.norm(rows[0]), 1.0, rtol=1e-5)


@pytest.mark.asyncio
async def test_bound_length_mismatch_raises():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    embedder.bind_dimension(2)
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(return_value=_resp([[1.0, 0.0, 0.0]])),
    ):
        with pytest.raises(IntelligenceError, match="Embedding length 3"):
            await embedder.aembed(["x"])


@pytest.mark.asyncio
async def test_order_preserved_one_to_one():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    embedder.bind_dimension(2)
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(return_value=_resp([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])),
    ):
        rows = await embedder.aembed(["a", "b", "c"])
    assert len(rows) == 3
    assert rows[0][1] == pytest.approx(0.0, abs=1e-6)
    assert rows[1][0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_ollama_prefix_and_custom_provider():
    embedder = OpenAICompatEmbedder(
        api_base="http://ollama", model="nomic", provider="ollama"
    )
    assert embedder.model_id == "ollama/nomic"
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(return_value=_resp([[1.0]])),
    ) as mock_emb:
        await embedder.aembed(["t"])
    assert mock_emb.call_args.kwargs["model"] == "ollama/nomic"
    assert mock_emb.call_args.kwargs["custom_llm_provider"] == "ollama"


@pytest.mark.asyncio
async def test_provider_omitted_when_unset():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(return_value=_resp([[1.0]])),
    ) as mock_emb:
        await embedder.aembed(["t"])
    assert "custom_llm_provider" not in mock_emb.call_args.kwargs


@pytest.mark.asyncio
async def test_timeout_and_auth_become_intelligence_error():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m", api_key="k")
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(side_effect=TimeoutError("timed out")),
    ):
        with pytest.raises(IntelligenceError):
            await embedder.aembed(["t"])
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(side_effect=RuntimeError("401 unauthorized")),
    ):
        with pytest.raises(IntelligenceError, match="401"):
            await embedder.aembed(["t"])


@pytest.mark.asyncio
async def test_semaphore_serializes_aembed():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    in_flight = 0
    max_in_flight = 0

    async def fake_aembedding(**kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return _resp([[1.0, 0.0]])

    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=fake_aembedding,
    ):
        await asyncio.gather(embedder.aembed(["a"]), embedder.aembed(["b"]))
    assert max_in_flight == 1


@pytest.mark.asyncio
async def test_close_is_noop_safe():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    await embedder.close()
    await embedder.close()


def test_unbound_dimension_raises():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    with pytest.raises(IntelligenceError, match="not bound"):
        _ = embedder.dimension


def test_bind_rejects_invalid_and_mismatch():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    with pytest.raises(IntelligenceError, match="positive int"):
        embedder.bind_dimension(0)
    embedder.bind_dimension(8)
    embedder.bind_dimension(8)
    with pytest.raises(IntelligenceError, match="already bound"):
        embedder.bind_dimension(4)


@pytest.mark.asyncio
async def test_aembed_rejects_unparseable_response():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    embedder.bind_dimension(2)
    resp = MagicMock()
    resp.data = [object()]
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(return_value=resp),
    ):
        with pytest.raises(IntelligenceError, match="Embedding request failed"):
            await embedder.aembed(["x"])


@pytest.mark.asyncio
async def test_aembed_count_mismatch():
    embedder = OpenAICompatEmbedder(api_base="http://e", model="m")
    embedder.bind_dimension(2)
    with patch(
        "code_rag.intelligence.openai_embedder.litellm.aembedding",
        new=AsyncMock(return_value=_resp([[1.0, 0.0], [0.0, 1.0]])),
    ):
        with pytest.raises(IntelligenceError, match="count mismatch"):
            await embedder.aembed(["only-one"])
