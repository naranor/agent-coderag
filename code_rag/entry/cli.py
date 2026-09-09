import logging
import asyncio
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
from ..core.utils import validate_path
from ..core.exceptions import CodeRAGError
from ..services import sync as sync_service
from ..services.discovery_api import run_api
from ..services.search import run_search
from ..services.setup import run_setup

logger = logging.getLogger(__name__)

# Legacy compatibility surface: these names are imported/patched by existing
# integrations and tests, so they must stay resolvable on this module.
__all__ = [
    "get_manager",
    "load_ignore_patterns",
    "should_index",
    "sync_cmd",
    "search_cmd",
    "api_cmd",
    "config_cmd",
    "setup_cmd",
    "rebuild_cmd",
    "DistillerConfig",
    "Embedder",
    "DuckDBStorage",
    "MultiParser",
    "Distiller",
    "get_global_dir",
    "requests",
    "validate_path",
    "main",
]


def load_ignore_patterns() -> pathspec.PathSpec:
    """Loads ignore patterns from the current directory's .gitignore or defaults."""
    return sync_service.load_ignore_patterns(Path.cwd())


def should_index(path: Path, ignore_spec: Optional[pathspec.PathSpec] = None) -> bool:
    """Filters files that should NOT be indexed."""
    return sync_service.should_index(path, ignore_spec)


async def sync_cmd(args):
    if args.path:
        args.path = str(validate_path(args.path))

    manager = get_manager(
        args.db,
        args.onnx,
        allow_build_execution=getattr(args, "allow_build_execution", False),
    )
    try:
        await sync_service.run_sync(
            manager,
            root=Path.cwd(),
            path=args.path,
            index_all=bool(args.all),
            force=bool(args.force),
        )

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
        results = await run_search(manager, args.query, limit=args.limit)

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
        report = await run_api(manager, args.library, lang=args.lang)

        if args.json:
            print(json.dumps({"library": report.library, "report": report.report}))
        else:
            print(report.report)
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

    if not args.json:
        print(f"Setting up agent-coderag in {global_dir}...")

    def _on_progress(event: str, name: str, error: Optional[BaseException]) -> None:
        if args.json:
            return
        if event == "skipped":
            print(f"  {name} already exists. Skipping.")
        elif event == "downloading":
            print(f"  Downloading {name}...")
        elif event == "error":
            print(f"  Error downloading {name}: {error}")

    try:
        # Pass module-level requests.get so @patch("code_rag.entry.cli.requests.get")
        # continues to intercept downloads in legacy CLI tests.
        await run_setup(
            force=args.force,
            global_dir=global_dir,
            on_progress=_on_progress,
            http_get=requests.get,
        )
    except Exception:
        # Legacy behaviour: per-file error already reported via on_progress;
        # stop here without crashing or printing "Setup complete.".
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


def get_manager(
    db_path: str,
    onnx_path: Optional[str] = None,
    allow_build_execution: bool = False,
) -> CodeRAGManager:
    """Wires a manager from the module-level components (legacy patch points).

    Mirrors ``code_rag.services.factory.create_manager`` but resolves the
    components through this module so existing CLI tests can substitute them.
    """
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
    sync.add_argument(
        "--allow-build-execution",
        action="store_true",
        help="Execute repository Maven/Gradle build files during dependency sync (trusted projects only).",
    )

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
