import logging
import asyncio
import sys
import json
from pathlib import Path
from typing import Optional

import requests
import pathspec

from ..api.client import CodeRAG
from ..api.models import SyncResult
from ..core.constants import DEFAULT_CONNECT_TIMEOUT_SECONDS
from ..core.exceptions import CodeRAGError
from ..core.utils import validate_path
from ..intelligence.distiller import Distiller, DistillerConfig
from ..intelligence.embedder import Embedder, get_global_dir
from ..parsers.multi_parser import MultiParser
from ..services import sync as sync_service
from ..services.config import apply_config_updates
from ..services.setup import run_setup
from ..storage.duckdb_impl import DuckDBStorage

logger = logging.getLogger(__name__)

# Legacy compatibility surface: these names are imported/patched by existing
# integrations and tests, so they must stay resolvable on this module.
__all__ = [
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
]


def _coderag_from_args(args) -> CodeRAG:
    return CodeRAG(
        db=args.db,
        onnx=getattr(args, "onnx", None),
        root=Path.cwd(),
        connect_timeout_seconds=float(
            getattr(args, "connect_timeout", DEFAULT_CONNECT_TIMEOUT_SECONDS)
        ),
        allow_build_execution=bool(getattr(args, "allow_build_execution", False)),
    )


def _emit_json_error(exc: Exception) -> None:
    payload = {"status": "error", "message": str(exc)}
    code = getattr(exc, "code", None)
    if isinstance(exc, CodeRAGError) and code is not None:
        payload["code"] = code.value if hasattr(code, "value") else str(code)
    print(json.dumps(payload))


def load_ignore_patterns() -> pathspec.PathSpec:
    """Loads ignore patterns from the current directory's .gitignore or defaults."""
    return sync_service.load_ignore_patterns(Path.cwd())


def should_index(
    path: Path,
    ignore_spec: Optional[pathspec.PathSpec] = None,
    *,
    root: Optional[Path] = None,
) -> bool:
    """Filters files that should NOT be indexed."""
    return sync_service.should_index(path, ignore_spec, root=root)


def _emit_sync_outcome(result: SyncResult, *, json_mode: bool, label: str) -> None:
    if result.status == "success":
        if json_mode:
            print(json.dumps({"status": "success", "indexed_files": "auto"}))
        return
    errors = [{"file": err.file, "message": err.message} for err in result.errors]
    if json_mode:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": f"{len(errors)} file(s) failed",
                    "errors": errors,
                }
            )
        )
    else:
        print(f"{label} completed with errors.", file=sys.stderr)
    raise SystemExit(1)


async def sync_cmd(args):
    if args.path:
        args.path = str(validate_path(args.path))

    rag = _coderag_from_args(args)
    try:
        result = await rag.sync(
            path=args.path,
            index_all=bool(args.all),
            force=bool(args.force),
        )
        _emit_sync_outcome(result, json_mode=args.json, label="Sync")
    except Exception as e:
        logger.error("Sync failed: %s", e)
        if args.json:
            _emit_json_error(e)
        else:
            print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    finally:
        await rag.close()


async def search_cmd(args):
    rag = _coderag_from_args(args)
    try:
        results = await rag.search(args.query, limit=args.limit)

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
    except Exception as e:
        logger.error("Search failed: %s", e)
        if args.json:
            _emit_json_error(e)
        else:
            print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    finally:
        await rag.close()


async def api_cmd(args):
    rag = _coderag_from_args(args)
    try:
        report = await rag.api(args.library, lang=args.lang)

        if args.json:
            print(json.dumps({"library": report.library, "report": report.report}))
        else:
            print(report.report)
    except Exception as e:
        logger.error("API discovery failed: %s", e)
        if args.json:
            _emit_json_error(e)
        else:
            print(f"Error: {e}", file=sys.stderr)
    finally:
        await rag.close()


DISTILLER_KEYS = ("model", "api_base", "api_key", "provider", "temperature")
EMBEDDING_KEYS = (
    "embedding_base",
    "embedding_key",
    "embedding_model",
    "embedding_provider",
)


def config_cmd(args):
    config = DistillerConfig.load()
    embedding_url = getattr(args, "embedding_url", None)
    embedding_key = getattr(args, "embedding_key", None)
    embedding_model = getattr(args, "embedding_model", None)
    embedding_provider = getattr(args, "embedding_provider", None)
    clear_embedding = bool(getattr(args, "clear_embedding", False))
    is_update = any(
        [
            args.url is not None,
            args.key is not None,
            args.model is not None,
            args.provider is not None,
            embedding_url is not None,
            embedding_key is not None,
            embedding_model is not None,
            embedding_provider is not None,
            clear_embedding,
        ]
    )
    if not is_update:
        dump = config.model_dump()
        if args.json:
            print(json.dumps(dump, indent=2))
        else:
            print("Distiller:")
            for key in DISTILLER_KEYS:
                print(f"  {key}: {dump.get(key)}")
            print("Embedding:")
            for key in EMBEDDING_KEYS:
                print(f"  {key}: {dump.get(key)}")
        return
    config = apply_config_updates(
        config,
        url=args.url,
        key=args.key,
        model=args.model,
        provider=args.provider,
        embedding_url=embedding_url,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        clear_embedding=clear_embedding,
    )
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
    rag = _coderag_from_args(args)
    try:
        result = await rag.rebuild()
        _emit_sync_outcome(result, json_mode=args.json, label="Rebuild")
    except Exception as exc:
        logger.error("Rebuild failed: %s", exc)
        if args.json:
            _emit_json_error(exc)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        await rag.close()


def dispatch(args) -> None:
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
