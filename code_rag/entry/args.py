"""Argparse entry. Importing this module must stay free of DuckDB, ONNX, and LiteLLM."""

import argparse
import importlib
import logging
import sys

from code_rag.core.constants import DEFAULT_CONNECT_TIMEOUT_SECONDS
from code_rag.core.exceptions import CodeRAGError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodeRAG: API Knowledge Bridge.")
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "DuckDB index file. Default: resolve legacy code_rag.db in cwd/root, "
            "else use .coderag.db under project root."
        ),
    )
    parser.add_argument("--onnx", help="Path to local ONNX model file.")
    parser.add_argument(
        "--connect-timeout",
        dest="connect_timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for a DuckDB file lock before StorageBusyError "
            f"(default {DEFAULT_CONNECT_TIMEOUT_SECONDS:g}). 0 = no retry."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON.")

    subparsers = parser.add_subparsers(dest="command")

    sync = subparsers.add_parser("sync", help="Index code units.")
    sync.add_argument("path", nargs="?", help="File or directory to index.")
    sync.add_argument("--all", action="store_true", help="Index all supported files.")
    sync.add_argument("--force", action="store_true", help="Force re-distillation.")
    sync.add_argument(
        "--allow-build-execution",
        action="store_true",
        help="Execute repository Maven/Gradle build files during dependency sync (trusted projects only).",
    )

    search = subparsers.add_parser("search", help="Semantic search.")
    search.add_argument("query", help="Natural language query.")
    search.add_argument("--limit", type=int, default=5, help="Result limit.")
    search.add_argument(
        "--relative-paths",
        dest="relative_paths",
        action="store_true",
        default=None,
        help="Print paths relative to the project root. Default: absolute paths under the current root.",
    )

    api = subparsers.add_parser("api", help="Discover library API.")
    api.add_argument("library", help="Library name (e.g., pydantic).")
    api.add_argument("--lang", help="Target language.")

    cfg = subparsers.add_parser("config", help="Manage LLM configuration.")
    cfg.add_argument("--url", help="API base URL.")
    cfg.add_argument("--key", help="API key.")
    cfg.add_argument("--model", help="Model name.")
    cfg.add_argument("--provider", help="Provider name (openai, ollama).")
    cfg.add_argument(
        "--embedding-url", dest="embedding_url", help="Embedding API base URL."
    )
    cfg.add_argument("--embedding-key", dest="embedding_key", help="Embedding API key.")
    cfg.add_argument(
        "--embedding-model", dest="embedding_model", help="Remote embedding model id."
    )
    cfg.add_argument(
        "--embedding-provider",
        dest="embedding_provider",
        help="LiteLLM custom_llm_provider for embeddings.",
    )
    cfg.add_argument(
        "--clear-embedding",
        dest="clear_embedding",
        action="store_true",
        help="Clear remote embedding config (local MiniLM).",
    )

    setup = subparsers.add_parser("setup", help="Initial setup (download models).")
    setup.add_argument("--force", action="store_true", help="Force model redownload.")

    subparsers.add_parser("rebuild", help="Full re-index of the project.")
    return parser


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args()

        if args.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

        if not args.command:
            parser.print_help()
            return

        # Heavy stack (DuckDB, ONNX, parsers) loads only for a real subcommand.
        dispatch = importlib.import_module("code_rag.entry.cli").dispatch
        dispatch(args)
    except CodeRAGError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).error("Unexpected error: %s", exc)
        sys.exit(1)
