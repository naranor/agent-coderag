from pathlib import Path
from typing import Optional

from code_rag.core.interfaces import IEmbedder, IIntelligence, IParser, IStorage
from code_rag.core.manager import CodeRAGManager
from code_rag.intelligence.distiller import Distiller, DistillerConfig
from code_rag.intelligence.factory import create_embedder
from code_rag.parsers.multi_parser import MultiParser
from code_rag.storage.db_connection import AccessMode, open_db_connection


async def create_stack(
    onnx_path: Optional[str] = None,
) -> tuple[IEmbedder, MultiParser, Distiller]:
    """Build the process-scoped embedder/parser/distiller stack."""
    config = DistillerConfig.load()
    distiller = Distiller(config)
    embedder = await create_embedder(config, onnx_path=onnx_path)
    parser = MultiParser()
    return embedder, parser, distiller


def build_manager(
    storage: IStorage,
    parser: IParser,
    distiller: IIntelligence,
    *,
    allow_build_execution: bool = False,
) -> CodeRAGManager:
    """Wire an ephemeral manager around an already-open storage connection."""
    return CodeRAGManager(
        storage,
        parser,
        distiller,
        allow_build_execution=allow_build_execution,
    )


async def create_manager(
    db_path: str,
    onnx_path: Optional[str] = None,
    allow_build_execution: bool = False,
    wipe: bool = False,
) -> CodeRAGManager:
    """Internal one-shot helper. Prefer CodeRAG for process-scoped lifetime."""
    embedder, parser, distiller = await create_stack(onnx_path)
    try:
        storage = await open_db_connection(
            Path(db_path),
            embedder,
            mode=AccessMode.READ_WRITE,
            wipe=wipe,
        )
    except Exception:
        await embedder.close()
        raise
    return build_manager(
        storage,
        parser,
        distiller,
        allow_build_execution=allow_build_execution,
    )
