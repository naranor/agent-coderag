import logging
import asyncio
import os
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import requests
import pathspec

from ..core.manager import CodeRAGManager
from ..storage.duckdb_impl import DuckDBStorage
from ..parsers.multi_parser import MultiParser
from ..intelligence.distiller import Distiller, DistillerConfig
from ..intelligence.embedder import Embedder, get_global_dir
from ..parsers.languages import EXTENSION_TO_LANGUAGE
from ..core.utils import validate_path
from ..core.exceptions import CodeRAGError, DiscoveryError

logger = logging.getLogger(__name__)


def load_ignore_patterns() -> pathspec.PathSpec:
    """Loads ignore patterns from .gitignore or defaults."""
    lines = []
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r", encoding="utf-8") as f:
            lines = f.readlines()

    common_excludes = [
        "node_modules/",
        "venv/",
        ".venv/",
        "__pycache__/",
        ".git/",
        ".idea/",
        ".vscode/",
        ".onnx",
        ".db",
        ".ai/",
    ]
    return pathspec.PathSpec.from_lines("gitignore", lines + common_excludes)


def should_index(path: Path, ignore_spec: Optional[pathspec.PathSpec] = None) -> bool:
    """Filters files that should NOT be indexed."""
    # Convert backslashes to forward slashes for cross-platform matching
    path_str = str(path).replace(os.sep, "/")

    # 1. Check against ignore spec (including .gitignore and common excludes)
    if ignore_spec and ignore_spec.match_file(path_str):
        return False

    return path.suffix.lower() in EXTENSION_TO_LANGUAGE


async def sync_cmd(args):
    # Validate input path
    if args.path:
        args.path = str(validate_path(args.path))

    manager = get_manager(args.db, args.onnx)
    try:
        ignore_spec = load_ignore_patterns()

        # Task 2: Sync dependencies before indexing
        try:
            await manager.sync_dependencies(args.path or ".")
        except DiscoveryError as de:
            logger.warning("Dependency discovery failed: %s", de)

        if args.path:
            target_path = Path(args.path)
            if target_path.is_file():
                if should_index(target_path, ignore_spec):
                    await manager.sync_file(str(target_path), force_distill=args.force)
            else:
                paths = [
                    str(p)
                    for p in target_path.rglob("*")
                    if p.is_file() and should_index(p, ignore_spec)
                ]

                if args.verbose:
                    logger.info("Indexing %d files...", len(paths))
                await manager.sync_project(paths, force_distill=args.force)
        elif args.all:
            paths = [
                str(p)
                for p in Path(".").rglob("*")
                if p.is_file() and should_index(p, ignore_spec)
            ]

            if args.verbose:
                logger.info("Indexing %d files...", len(paths))
            await manager.sync_project(paths, force_distill=args.force)

        if args.json:
            print(json.dumps({"status": "success", "indexed_files": "auto"}))
    except Exception as e:
        logger.error("Sync failed: %s", e)
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
    finally:
        await manager.close()


async def search_cmd(args):
    manager = get_manager(args.db, args.onnx)
    try:
        results = await manager.search(args.query, limit=args.limit)

        if args.json:
            output = []
            for r in results:
                output.append(
                    {
                        "id": r.id,
                        "name": r.name,
                        "kind": r.kind.value,
                        "path": r.path,
                        "signature": r.signature,
                        "summary": r.summary,
                    }
                )
            print(json.dumps(output, indent=2))
        else:
            if not results:
                print("No results.")
                return

            print(f"Search results for: '{args.query}'\n")
            for r in results:
                print(f"[{r.kind.value}] {r.name} ({r.path})")
                if r.summary:
                    print(f"  Summary: {r.summary}")
                print("-" * 20)
    finally:
        await manager.close()


async def api_cmd(args):
    manager = get_manager(args.db, args.onnx)
    try:
        # Default to python if not specified
        lang = args.lang or "python"
        api_report = await manager.discovery.extract_api(args.library, language=lang)

        if args.json:
            print(json.dumps({"library": args.library, "report": api_report}))
        else:
            print(api_report)
    except Exception as e:
        logger.error("API discovery failed: %s", e)
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
    finally:
        await manager.close()


def config_cmd(args):
    config = DistillerConfig.load()

    # If no args, just show current config
    if not (args.url or args.key or args.model or args.provider):
        if args.json:
            print(json.dumps(config.model_dump(), indent=2))
        else:
            print("Current configuration:")
            for k, v in config.model_dump().items():
                print(f"  {k}: {v}")
        return

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
        print(json.dumps({"status": "success", "message": "Config updated"}))
    else:
        print("Config updated.")


async def setup_cmd(args):
    """Downloads necessary local models."""
    global_dir = get_global_dir()
    model_dir = global_dir / "models" / "mini-lm"
    model_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "model.onnx": "https://huggingface.co/naranor/all-MiniLM-L6-v2-onnx/resolve/main/model.onnx",
        "tokenizer.json": "https://huggingface.co/naranor/all-MiniLM-L6-v2-onnx/resolve/main/tokenizer.json",
    }

    if not args.json:
        print(f"Setting up agent-coderag in {global_dir}...")

    for name, url in files.items():
        target_path = model_dir / name
        if target_path.exists() and not args.force:
            if not args.json:
                print(f"  {name} already exists. Skipping.")
            continue

        if not args.json:
            print(f"  Downloading {name}...")
        tmp_path = target_path.with_suffix(".tmp")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.replace(tmp_path, target_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            if not args.json:
                print(f"  Error downloading {name}: {e}")
            return

    if not args.json:
        print("Setup complete.")


async def rebuild_cmd(args):
    """Full re-index of the current project."""
    # Force re-distill all files
    args.all = True
    args.force = True
    args.path = None
    await sync_cmd(args)


def get_manager(db_path: str, onnx_path: Optional[str] = None) -> CodeRAGManager:
    # 1. Setup Intelligence
    config = DistillerConfig.load()
    distiller = Distiller(config)
    embedder = Embedder(model_path=onnx_path)

    # 2. Setup Storage
    storage = DuckDBStorage(db_path, embedder=embedder)

    # 3. Setup Parser
    parser = MultiParser()

    return CodeRAGManager(storage, parser, distiller)


def main():
    parser = argparse.ArgumentParser(description="CodeRAG: API Knowledge Bridge.")
    parser.add_argument(
        "--db", default="code_rag.db", help="Path to DuckDB database file."
    )
    parser.add_argument("--onnx", help="Path to local ONNX model file.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON.")

    subparsers = parser.add_subparsers(dest="command")

    # Sync
    sync = subparsers.add_parser("sync", help="Index code units.")
    sync.add_argument("path", nargs="?", help="File or directory to index.")
    sync.add_argument("--all", action="store_true", help="Index all supported files.")
    sync.add_argument("--force", action="store_true", help="Force re-distillation.")

    # Search
    search = subparsers.add_parser("search", help="Semantic search.")
    search.add_argument("query", help="Natural language query.")
    search.add_argument("--limit", type=int, default=5, help="Result limit.")

    # API
    api = subparsers.add_parser("api", help="Discover library API.")
    api.add_argument("library", help="Library name (e.g., pydantic).")
    api.add_argument("--lang", help="Target language.")

    # Config
    cfg = subparsers.add_parser("config", help="Manage LLM configuration.")
    cfg.add_argument("--url", help="API base URL.")
    cfg.add_argument("--key", help="API key.")
    cfg.add_argument("--model", help="Model name.")
    cfg.add_argument("--provider", help="Provider name (openai, ollama).")

    # Setup
    setup = subparsers.add_parser("setup", help="Initial setup (download models).")
    setup.add_argument("--force", action="store_true", help="Force model redownload.")

    # Rebuild
    subparsers.add_parser("rebuild", help="Full re-index of the project.")

    try:
        args = parser.parse_args()

        if args.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

        if args.command == "sync":
            asyncio.run(sync_cmd(args))
        elif args.command == "search":
            asyncio.run(search_cmd(args))
        elif args.command == "api":
            asyncio.run(api_cmd(args))
        elif args.command == "config":
            config_cmd(args)
        elif args.command == "setup":
            asyncio.run(setup_cmd(args))
        elif args.command == "rebuild":
            asyncio.run(rebuild_cmd(args))
        else:
            parser.print_help()
    except CodeRAGError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
