import asyncio
import argparse
import os
import sys
import logging
import json
import fnmatch
from pathlib import Path
from typing import Optional, List
import httpx

# Suppress external library noise before they are imported by other modules
os.environ["LITELLM_VERBOSE"] = "FALSE"
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# pylint: disable=wrong-import-position
from ..core.manager import CodeRAGManager
from ..storage.duckdb_impl import DuckDBStorage
from ..parsers.ast_index import AstIndexParser
from ..intelligence.embedder import Embedder, get_default_model_dir
from ..intelligence.distiller import Distiller, DistillerConfig
from ..discovery.dependency import extract_library_api
# pylint: enable=wrong-import-position

# Setup basic logging - default to WARNING for clean output
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("agent-coderag")

def get_manager(db_path: str, onnx_path: Optional[str] = None, verbose: bool = False):
    """Initializes the RAG manager."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        
    config = DistillerConfig.load()
    config.api_base = os.getenv("AGENT_PROXY_URL", config.api_base)
    config.api_key = os.getenv("AGENT_PROXY_KEY", config.api_key)
    config.model = os.getenv("AGENT_MODEL", config.model)
    config.provider = os.getenv("AGENT_PROVIDER", config.provider)
    
    embedder = Embedder(model_path=onnx_path)
    storage = DuckDBStorage(db_path, embedder=embedder)
    parser = AstIndexParser()
    distiller = Distiller(config)
    
    return CodeRAGManager(storage, parser, distiller)

def load_ignore_patterns() -> List[str]:
    """Loads patterns from .gitignore if it exists."""
    patterns = []
    ignore_file = Path(".gitignore")
    if ignore_file.exists():
        try:
            with open(ignore_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except Exception as e:
            logger.warning("Failed to load .gitignore: %s", e)
    return patterns

def should_index(path: Path, ignore_patterns: Optional[List[str]] = None) -> bool:
    """Filters files that should NOT be indexed."""
    # 1. Default hardcoded exclusions
    parts = set(path.parts)
    exclude_dirs = {'tests', 'venv', '__pycache__', '.git', '.pytest_cache', 'dist', 'build'}
    if any(ex in parts for ex in exclude_dirs):
        return False
    
    # 2. Check .gitignore patterns
    if ignore_patterns:
        p_str = str(path).replace(os.sep, '/')
        for pattern in ignore_patterns:
            # Simple fnmatch support for gitignore-like patterns
            if fnmatch.fnmatch(p_str, pattern) or fnmatch.fnmatch(p_str, f"*/{pattern}"):
                return False

    return path.suffix == '.py'

async def sync_cmd(args):
    manager = get_manager(args.db, args.onnx, args.verbose)
    ignore_patterns = load_ignore_patterns()
    
    if args.path:
        target_path = Path(args.path)
        if target_path.is_file():
            await manager.sync_file(str(target_path), force_distill=args.force)
        else:
            paths = [str(p) for p in target_path.rglob("*.py") if should_index(p, ignore_patterns)]
            if args.verbose:
                logger.info("Indexing %d files...", len(paths))
            await manager.sync_project(paths, force_distill=args.force)
    elif args.all:
        paths = [str(p) for p in Path(".").rglob("*.py") if should_index(p, ignore_patterns)]
        if args.verbose:
            logger.info("Indexing %d files...", len(paths))
        await manager.sync_project(paths, force_distill=args.force)
    
    if not args.json:
        print("Done.")
    else:
        print(json.dumps({"status": "success"}))

async def search_cmd(args):
    manager = get_manager(args.db, args.onnx, args.verbose)
    results = await manager.search(args.query, limit=args.limit)
    
    if args.json:
        output = [unit.model_dump() for unit in results]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if not results:
        print("No results.")
        return

    for unit in results:
        print(f"[{unit.kind.value}] {unit.name} | {unit.path}")
        if unit.summary:
            print(f"  {unit.summary}")
        print("-" * 20)

async def api_cmd(args):
    output = await extract_library_api(args.library)
    if args.json:
        print(json.dumps({"api": output}))
    else:
        print(output)

async def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                return False
            with open(dest, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
    return True

async def setup_cmd(args):
    dest_dir = get_default_model_dir()
    base_url = "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"
    files = {
        "model.onnx": f"{base_url}/onnx/model.onnx",
        "tokenizer.json": f"{base_url}/tokenizer.json"
    }
    force = getattr(args, "force", False)
    for filename, url in files.items():
        dest_path = dest_dir / filename
        if not dest_path.exists() or force:
            await download_file(url, dest_path)
    print("Setup complete.")

def config_cmd(args):
    config = DistillerConfig.load()
    if args.url:
        config.api_base = args.url
    if args.key:
        config.api_key = args.key
    if args.model:
        config.model = args.model
    if args.provider:
        config.provider = args.provider
    config.save()
    if args.json:
        print(json.dumps(config.model_dump()))
    else:
        print("Config updated.")

async def rebuild_cmd(args):
    db_path = Path(args.db)
    if db_path.exists():
        if args.verbose:
            logger.info("Removing old database: %s", db_path)
        db_path.unlink()
    
    # Trigger full sync
    args.all = True
    args.force = True
    args.path = None
    await sync_cmd(args)


def main():
    parser = argparse.ArgumentParser(description="Agent-CodeRAG CLI Tool")
    parser.add_argument("--db", default=".code_rag.db", help="Path to DuckDB database")
    parser.add_argument("--onnx", help="Path to ONNX embedding model")
    parser.add_argument("--verbose", action="store_true", help="Show debug logs")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Commands
    subparsers.add_parser("setup", help="Download models")
    subparsers.add_parser("rebuild", help="Nuke database and re-index everything")
    
    conf_p = subparsers.add_parser("config", help="AI settings")
    conf_p.add_argument("--url", help="API base URL")
    conf_p.add_argument("--key", help="API key")
    conf_p.add_argument("--model", help="Model name")
    conf_p.add_argument("--provider", help="Provider")

    sync_p = subparsers.add_parser("sync", help="Index files")
    sync_p.add_argument("path", nargs="?", help="Path to index")
    sync_p.add_argument("--all", action="store_true", help="Index all python files")
    sync_p.add_argument("--force", action="store_true", help="Force distillation")

    search_p = subparsers.add_parser("search", help="Semantic search")
    search_p.add_argument("query", help="Query")
    search_p.add_argument("--limit", type=int, default=5, help="Max results")

    api_p = subparsers.add_parser("api", help="Discover library API")
    api_p.add_argument("library", help="Library name")

    args = parser.parse_args()

    try:
        if args.command == "setup":
            asyncio.run(setup_cmd(args))
        elif args.command == "config":
            config_cmd(args)
        elif args.command == "rebuild":
            asyncio.run(rebuild_cmd(args))
        elif args.command == "sync":
            asyncio.run(sync_cmd(args))
        elif args.command == "search":
            asyncio.run(search_cmd(args))
        elif args.command == "api":
            asyncio.run(api_cmd(args))
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
