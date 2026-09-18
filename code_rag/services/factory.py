from typing import Optional

from code_rag.intelligence.distiller import Distiller, DistillerConfig
from code_rag.intelligence.factory import create_embedder
from code_rag.parsers.multi_parser import MultiParser


async def create_stack(
    onnx_path: Optional[str] = None,
) -> tuple:
    """Build the process-scoped embedder/parser/distiller stack."""
    config = DistillerConfig.load()
    distiller = Distiller(config)
    embedder = await create_embedder(config, onnx_path=onnx_path)
    parser = MultiParser()
    return embedder, parser, distiller
