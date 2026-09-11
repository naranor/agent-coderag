import logging
import os
from pathlib import Path
from typing import Optional

import pathspec

from code_rag.api.models import SyncFileError, SyncResult
from code_rag.core.exceptions import DiscoveryError
from code_rag.core.utils import validate_path
from code_rag.parsers.languages import EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)


def load_ignore_patterns(root: Path) -> pathspec.PathSpec:
    """Loads ignore patterns from root's .gitignore or defaults."""
    lines = []
    gitignore_path = root / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as f:
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
    path_str = str(path).replace(os.sep, "/")

    if ignore_spec and ignore_spec.match_file(path_str):
        return False

    return path.suffix.lower() in EXTENSION_TO_LANGUAGE


def _result_from_failures(
    indexed_files: int, failures: list[tuple[str, str]]
) -> SyncResult:
    errors = [SyncFileError(file=path, message=message) for path, message in failures]
    status = "error" if errors else "success"
    return SyncResult(status=status, indexed_files=indexed_files, errors=errors)


async def run_sync(
    manager,
    *,
    root: Path,
    path: Optional[str] = None,
    index_all: bool = False,
    force: bool = False,
) -> SyncResult:
    """Syncs a single path, a directory tree, or the whole project into the index."""
    if path is None and not index_all:
        return SyncResult(status="success", indexed_files=0)

    validated_path = str(validate_path(path, root=root)) if path else None

    try:
        await manager.sync_dependencies(validated_path or str(root))
    except DiscoveryError as de:
        logger.warning("Dependency discovery failed: %s", de)

    ignore_spec = load_ignore_patterns(root)
    indexed_files = 0
    failures: list[tuple[str, str]] = []

    if validated_path:
        target_path = Path(validated_path)
        if target_path.is_file():
            if should_index(target_path, ignore_spec):
                failures = await manager.sync_project(
                    [str(target_path)], force_distill=force, index_all=index_all
                )
                indexed_files = 1
        else:
            paths = [
                str(p)
                for p in target_path.rglob("*")
                if p.is_file() and should_index(p, ignore_spec)
            ]
            if paths or index_all:
                failures = await manager.sync_project(
                    paths, force_distill=force, index_all=index_all
                )
            indexed_files = len(paths)
    elif index_all:
        paths = [
            str(p)
            for p in root.rglob("*")
            if p.is_file() and should_index(p, ignore_spec)
        ]
        failures = await manager.sync_project(
            paths, force_distill=force, index_all=index_all
        )
        indexed_files = len(paths)

    return _result_from_failures(indexed_files, failures)


async def run_rebuild(manager, *, root: Path) -> SyncResult:
    """Forces a full re-index of the entire project."""
    return await run_sync(manager, root=root, path=None, index_all=True, force=True)
