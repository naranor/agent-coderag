import asyncio
import sys
from typing import Optional

from ..core.exceptions import IntelligenceError
from ..core.interfaces import IEmbedder
from .distiller import DistillerConfig
from .embedder import LocalOnnxEmbedder
from .openai_embedder import OpenAICompatEmbedder


async def create_embedder(
    config: DistillerConfig,
    onnx_path: Optional[str] = None,
) -> IEmbedder:
    base = config.embedding_base
    model = config.embedding_model
    if (base is None) != (model is None):
        raise IntelligenceError(
            "embedding_base and embedding_model must both be set or both unset"
        )
    if base and model:
        if onnx_path:
            print(
                "Warning: --onnx is ignored because remote embeddings are configured.",
                file=sys.stderr,
            )
        return OpenAICompatEmbedder(
            api_base=base,
            model=model,
            api_key=config.embedding_key,
            provider=config.embedding_provider,
        )
    return await asyncio.to_thread(LocalOnnxEmbedder, model_path=onnx_path)
