from typing import Optional

from code_rag.core.manager import CodeRAGManager
from code_rag.intelligence.distiller import Distiller, DistillerConfig
from code_rag.intelligence.factory import create_embedder
from code_rag.parsers.multi_parser import MultiParser
from code_rag.storage.duckdb_impl import DuckDBStorage


async def create_manager(
    db_path: str,
    onnx_path: Optional[str] = None,
    allow_build_execution: bool = False,
    wipe: bool = False,
) -> CodeRAGManager:
    config = DistillerConfig.load()
    distiller = Distiller(config)
    embedder = await create_embedder(config, onnx_path=onnx_path)
    try:
        storage = await DuckDBStorage.open(db_path, embedder, wipe=wipe)
    except Exception:
        await embedder.close()
        raise
    parser = MultiParser()
    return CodeRAGManager(
        storage,
        parser,
        distiller,
        allow_build_execution=allow_build_execution,
    )
