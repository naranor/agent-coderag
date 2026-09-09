from typing import Optional

from code_rag.core.manager import CodeRAGManager
from code_rag.intelligence.distiller import Distiller, DistillerConfig
from code_rag.intelligence.embedder import Embedder
from code_rag.parsers.multi_parser import MultiParser
from code_rag.storage.duckdb_impl import DuckDBStorage


def create_manager(
    db_path: str,
    onnx_path: Optional[str] = None,
    allow_build_execution: bool = False,
) -> CodeRAGManager:
    """Builds a fully wired CodeRAGManager (storage, parser, distiller)."""
    config = DistillerConfig.load()
    distiller = Distiller(config)
    embedder = Embedder(model_path=onnx_path)

    storage = DuckDBStorage(db_path, embedder=embedder)
    parser = MultiParser()

    return CodeRAGManager(
        storage,
        parser,
        distiller,
        allow_build_execution=allow_build_execution,
    )
