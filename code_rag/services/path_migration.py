"""One-time rewrite of absolute index paths onto project-relative posix paths."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from code_rag.core.models import KnowledgeUnit
from code_rag.parsers.tree_sitter import GrammarNotFoundError
from code_rag.paths import project_relative_posix
from code_rag.storage.duckdb_impl import DuckDBStorage

logger = logging.getLogger(__name__)


def _groups(units: list[KnowledgeUnit]) -> dict[str, Counter[str]]:
    groups: dict[str, Counter[str]] = {}
    for unit in units:
        if not Path(unit.path).is_absolute():
            continue
        groups.setdefault(unit.path, Counter())[unit.code_hash] += 1
    return groups


def claim_relative_paths(
    groups: dict[str, Counter[str]],
    root: Path,
    files: list[tuple[Path, Counter[str]]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    pending: list[tuple[str, Counter[str]]] = []
    for old, hashes in groups.items():
        path = Path(old)
        if path.is_absolute():
            try:
                rel = project_relative_posix(path, root)
            except ValueError:
                pending.append((old, hashes))
                continue
            if rel in taken:
                continue
            mapping[old] = rel
            taken.add(rel)
            continue
        pending.append((old, hashes))

    claimed: set[str] = set()
    for file_path, hashes in files:
        try:
            rel = project_relative_posix(file_path, root)
        except ValueError:
            continue
        if rel in taken:
            continue
        for old, group_hashes in pending:
            if old in claimed or old in mapping:
                continue
            if group_hashes != hashes:
                continue
            mapping[old] = rel
            taken.add(rel)
            claimed.add(old)
            break
    return mapping


def _suffix(path: str) -> str:
    return Path(path).suffix.lower()


async def migrate_absolute_paths(
    storage, parser, root: Path, files: list[Path]
) -> bool:
    if not isinstance(storage, DuckDBStorage):
        return False
    if await storage.paths_migration_done():
        return False
    groups = _groups(await storage.list_units())
    if not groups:
        await storage.commit_path_migration({})
        return False
    incomplete = False
    skipped_suffixes: set[str] = set()
    parsed: list[tuple[Path, Counter[str]]] = []
    for path in files:
        try:
            units = await parser.distill_file(str(path), raise_on_failure=True)
        except GrammarNotFoundError:
            logger.warning("Skipping unparsed file during path migration: %s", path)
            skipped_suffixes.add(_suffix(str(path)))
            incomplete = True
            continue
        except Exception:
            logger.exception("Path migration could not parse %s", path)
            skipped_suffixes.add(_suffix(str(path)))
            incomplete = True
            continue
        parsed.append((path, Counter(unit.code_hash for unit in units)))
    mapping = claim_relative_paths(groups, root, parsed)
    failed_commits: set[str] = set()
    for old, new in mapping.items():
        try:
            await storage.commit_rewritten_path(old, new)
        except Exception:
            logger.exception("Path migration failed for %s", old)
            failed_commits.add(old)
            incomplete = True
    orphans = [
        old
        for old in groups
        if old not in mapping
        and old not in failed_commits
        and _suffix(old) not in skipped_suffixes
    ]
    if orphans:
        await storage.delete_absolute_paths(orphans)
    if not incomplete:
        await storage.mark_paths_migrated()
    return True
