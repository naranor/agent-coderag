import asyncio
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List

from ..core.manager import CodeRAGManager
from ..storage.duckdb_impl import DuckDBStorage
from ..parsers.ast_index import AstIndexParser
from ..intelligence.embedder import Embedder
from ..intelligence.distiller import Distiller, DistillerConfig

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("code-rag-cli")

def get_manager(db_path: str, onnx_path: str = None):
    """Initializes the RAG manager with default components."""
    # Fallback to environment or default path if onnx_path not provided
    onnx_path = onnx_path or os.getenv("RAG_ONNX_PATH", "models/bge-small-en-v1.5.onnx")
    
    embedder = Embedder(model_path=onnx_path)
    storage = DuckDBStorage(db_path, embedder=embedder)
    parser = AstIndexParser()
    
    distiller_config = DistillerConfig(
        model=os.getenv("AGENT_MODEL", "auto"),
        api_base=os.getenv("AGENT_PROXY_URL", "http://localhost:8383/api/v1"),
        api_key=os.getenv("AGENT_PROXY_KEY", "sk-not-required"),
        provider="openai"
    )
    distiller = Distiller(distiller_config)
    
    return CodeRAGManager(storage, parser, distiller)

def should_index(path: Path) -> bool:
    """Filters files that should NOT be indexed."""
    p_str = str(path)
    
    # Negative constraints
    exclude_patterns = [
        "tests/", 
        "MagicMock/", 
        "venv/", 
        "__pycache__/", 
        "archive/", 
        "sessions/",
        "download_model.py",
        "index_core.py",
        "rag.py",
        ".git/"
    ]
    
    for pattern in exclude_patterns:
        if pattern in p_str:
            return False
            
    if path.suffix != ".py":
        return False
        
    return True

async def sync_cmd(args):
    manager = get_manager(args.db, args.onnx)
    
    if args.path:
        target_path = Path(args.path)
        if target_path.is_file():
            await manager.sync_file(str(target_path), force_distill=args.force)
        else:
            # Recursively find all .py files in directory
            paths = [str(p) for p in target_path.rglob("*.py") if should_index(p)]
            logger.info(f"Indexing {len(paths)} files from {args.path}...")
            await manager.sync_project(paths, force_distill=args.force)
    elif args.all:
        # Find all .py files in current directory
        paths = [str(p) for p in Path(".").rglob("*.py") if should_index(p)]
        logger.info(f"Indexing {len(paths)} files in current directory...")
        await manager.sync_project(paths, force_distill=args.force)
    else:
        logger.error("Please specify a path or use --all")
        sys.exit(1)
        
    logger.info("Sync complete.")

async def search_cmd(args):
    manager = get_manager(args.db, args.onnx)
    results = await manager.search(args.query, limit=args.limit)
    if not results:
        print("No results found.")
        return

    print(f"\n--- Results for '{args.query}' ---")
    for i, unit in enumerate(results, 1):
        print(f"{i}. [{unit.kind.value}] {unit.name} ({unit.path})")
        if unit.summary:
            print(f"   Intent: {unit.summary}")
        if unit.signature:
            print(f"   Signature: {unit.signature}")
        print("-" * 40)

async def api_cmd(args):
    from ..discovery.dependency import extract_library_api
    output = await extract_library_api(args.library)
    print(output)

def main():
    parser = argparse.ArgumentParser(description="CodeRAG CLI Tool")
    parser.add_argument("--db", default=".code_rag.db", help="Path to DuckDB database")
    parser.add_argument("--onnx", help="Path to ONNX embedding model")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sync command
    sync_p = subparsers.add_parser("sync", help="Synchronize files with knowledge base")
    sync_p.add_argument("path", nargs="?", help="Path to file or directory")
    sync_p.add_argument("--all", action="store_true", help="Sync all python files in current dir")
    sync_p.add_argument("--force", action="store_true", help="Force re-distillation")

    # Search command
    search_p = subparsers.add_parser("search", help="Semantic search in knowledge base")
    search_p.add_argument("query", help="Natural language query")
    search_p.add_argument("--limit", type=int, default=5, help="Max results")

    # API command
    api_p = subparsers.add_parser("api", help="Extract public API of an installed library")
    api_p.add_argument("library", help="Library name (e.g., pydantic_ai)")

    args = parser.parse_args()

    if args.command == "sync":
        asyncio.run(sync_cmd(args))
    elif args.command == "search":
        asyncio.run(search_cmd(args))
    elif args.command == "api":
        asyncio.run(api_cmd(args))

if __name__ == "__main__":
    main()
