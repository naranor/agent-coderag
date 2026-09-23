import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pathspec

from code_rag.api.models import SyncFileError, SyncResult
from code_rag.core.constants import MAX_CONCURRENT_TASKS
from code_rag.core.exceptions import DiscoveryError
from code_rag.core.utils import validate_path
from code_rag.paths import project_relative_posix
from code_rag.parsers.languages import EXTENSION_TO_LANGUAGE
from code_rag.services.dependencies import sync_dependencies
from code_rag.services.indexing import IndexStack, sync_project

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncOptions:
    """Parameters for a sync/rebuild use-case (no resource ownership)."""

    root: Path
    path: Optional[str] = None
    index_all: bool = False
    force: bool = False
    allow_build_execution: bool = False
    max_concurrency: int = MAX_CONCURRENT_TASKS


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


def should_index(
    path: Path,
    ignore_spec: Optional[pathspec.PathSpec] = None,
    *,
    root: Optional[Path] = None,
) -> bool:
    """Filters files that should NOT be indexed.

    When ``root`` is set, gitignore sees only the posix path relative to ``root``.
    """
    if root is not None:
        try:
            match_target = project_relative_posix(path, root)
        except ValueError:
            return False
    else:
        match_target = str(path).replace(os.sep, "/")

    if ignore_spec and ignore_spec.match_file(match_target):
        return False

    return path.suffix.lower() in EXTENSION_TO_LANGUAGE


def _result_from_failures(
    indexed_files: int, failures: list[tuple[str, str]]
) -> SyncResult:
    errors = [SyncFileError(file=path, message=message) for path, message in failures]
    status = "error" if errors else "success"
    return SyncResult(status=status, indexed_files=indexed_files, errors=errors)


def _indexable_under(
    walk_root: Path,
    ignore_spec: pathspec.PathSpec,
    *,
    project_root: Path,
) -> list[str]:
    return [
        str(p)
        for p in walk_root.rglob("*")
        if p.is_file() and should_index(p, ignore_spec, root=project_root)
    ]


async def _run_project_sync(
    stack: IndexStack,
    paths: list[str],
    options: SyncOptions,
) -> list[tuple[str, str]]:
    return await sync_project(
        stack,
        paths,
        force_distill=options.force,
        index_all=options.index_all,
        max_concurrency=options.max_concurrency,
    )


async def _sync_validated_path(
    stack: IndexStack,
    validated_path: str,
    options: SyncOptions,
    ignore_spec: pathspec.PathSpec,
) -> SyncResult:
    target_path = Path(validated_path)
    if target_path.is_file():
        if not should_index(target_path, ignore_spec, root=options.root):
            return _result_from_failures(0, [])
        failures = await _run_project_sync(stack, [str(target_path)], options)
        return _result_from_failures(1, failures)

    paths = _indexable_under(target_path, ignore_spec, project_root=options.root)
    if not paths and not options.index_all:
        return _result_from_failures(len(paths), [])
    failures = await _run_project_sync(stack, paths, options)
    return _result_from_failures(len(paths), failures)


async def run_sync(stack: IndexStack, options: SyncOptions) -> SyncResult:
    """Syncs a single path, a directory tree, or the whole project into the index."""
    if options.path is None and not options.index_all:
        return SyncResult(status="success", indexed_files=0)

    validated_path = (
        str(validate_path(options.path, root=options.root)) if options.path else None
    )

    try:
        await sync_dependencies(
            stack.storage,
            validated_path or str(options.root),
            allow_build_execution=options.allow_build_execution,
        )
    except DiscoveryError as de:
        logger.warning("Dependency discovery failed: %s", de)

    ignore_spec = load_ignore_patterns(options.root)

    if validated_path:
        return await _sync_validated_path(stack, validated_path, options, ignore_spec)

    paths = _indexable_under(options.root, ignore_spec, project_root=options.root)
    failures = await _run_project_sync(stack, paths, options)
    return _result_from_failures(len(paths), failures)


async def run_rebuild(
    stack: IndexStack,
    *,
    root: Path,
    allow_build_execution: bool = False,
    max_concurrency: int = MAX_CONCURRENT_TASKS,
) -> SyncResult:
    """Forces a full re-index of the entire project."""
    return await run_sync(
        stack,
        SyncOptions(
            root=root,
            path=None,
            index_all=True,
            force=True,
            allow_build_execution=allow_build_execution,
            max_concurrency=max_concurrency,
        ),
    )
