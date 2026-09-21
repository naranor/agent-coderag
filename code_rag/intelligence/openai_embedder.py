import asyncio
import importlib
import logging
from typing import Optional

import numpy as np

from ..core.constants import EMBEDDING_REQUEST_TIMEOUT
from ..core.exceptions import IntelligenceError
from ..core.interfaces import IEmbedder
from .embedder import l2_normalize_rows

logger = logging.getLogger(__name__)


def _vector_from_item(item: object) -> list[float]:
    if isinstance(item, dict):
        vec = item.get("embedding")
    else:
        vec = getattr(item, "embedding", None)
    if vec is None:
        raise TypeError("embedding item has no embedding")
    if isinstance(vec, (list, tuple)) and vec and isinstance(vec[0], (list, tuple)):
        vec = vec[0]
    return list(vec)


class OpenAICompatEmbedder(IEmbedder):
    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self._api_base = api_base
        self._api_key = api_key
        self._provider = provider
        self._model = model
        if provider == "ollama" and not model.startswith("ollama/"):
            self._model = f"ollama/{model}"
        self._dimension: Optional[int] = None
        self._embed_lock = asyncio.Semaphore(1)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise IntelligenceError("Embedding dimension is not bound")
        return self._dimension

    @property
    def model_id(self) -> str:
        return self._model

    def bind_dimension(self, dim: int) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise IntelligenceError("embedding dimension must be a positive int")
        if self._dimension is None:
            self._dimension = dim
            return
        if dim != self._dimension:
            raise IntelligenceError(
                f"Cannot bind dimension {dim}; embedder already bound to {self._dimension}"
            )

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        kwargs = {
            "model": self._model,
            "input": texts,
            "api_base": self._api_base,
            "api_key": self._api_key,
            "timeout": EMBEDDING_REQUEST_TIMEOUT,
        }
        if self._provider is not None:
            kwargs["custom_llm_provider"] = self._provider
        litellm = importlib.import_module("litellm")

        async with self._embed_lock:
            try:
                response = await litellm.aembedding(**kwargs)
            except Exception as exc:
                raise IntelligenceError(f"Embedding request failed: {exc}") from exc
        try:
            raw = [_vector_from_item(item) for item in response.data]
        except Exception as exc:
            raise IntelligenceError(f"Embedding request failed: {exc}") from exc
        if len(raw) != len(texts):
            raise IntelligenceError("Embedding count mismatch")
        if self._dimension is not None:
            for vec in raw:
                if len(vec) != self._dimension:
                    raise IntelligenceError(
                        f"Embedding length {len(vec)} does not match bound dimension {self._dimension}"
                    )
        matrix = l2_normalize_rows(np.asarray(raw, dtype=float))
        return matrix.tolist()

    async def close(self) -> None:
        return None
