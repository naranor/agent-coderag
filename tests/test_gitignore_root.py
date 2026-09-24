from pathlib import Path

import pytest

from code_rag.services.indexing import IndexStack
from code_rag.services.sync import (
    SyncOptions,
    load_ignore_patterns,
    run_sync,
    should_index,
)


def test_worktree_root_is_indexed_and_nested_worktree_is_ignored(tmp_path: Path):
    root = tmp_path / ".worktrees" / "feature"
    src = root / "src" / "a.py"
    nested = root / ".worktrees" / "other" / "x.py"
    src.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    src.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    nested.write_text("def beta():\n    return 1\n", encoding="utf-8")
    (root / ".gitignore").write_text(".worktrees/\n", encoding="utf-8")

    spec = load_ignore_patterns(root)
    absolute = str(src.resolve()).replace("\\", "/")
    assert spec.match_file(absolute) is True

    assert should_index(src, spec, root=root) is True
    assert should_index(nested, spec, root=root) is False


def test_path_outside_root_is_not_indexed(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "other" / "a.py"
    outside.parent.mkdir()
    outside.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    spec = load_ignore_patterns(root)
    assert should_index(outside, spec, root=root) is False


@pytest.mark.asyncio
async def test_directory_sync_matches_gitignore_against_project_root(
    tmp_path: Path, monkeypatch
):
    """`sync src` still applies the project-root `.gitignore`, not `src/` as the match root."""
    root = tmp_path / "proj"
    generated = root / "src" / "generated" / "a.py"
    kept = root / "src" / "kept.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    kept.write_text("def beta():\n    return 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("src/generated/\n", encoding="utf-8")

    seen: dict[str, list[str]] = {}

    async def fake_sync_project(stack, paths, **kwargs):
        seen["paths"] = list(paths)
        return []

    monkeypatch.setattr("code_rag.services.sync.sync_project", fake_sync_project)
    result = await run_sync(
        IndexStack(object(), object(), object()),
        SyncOptions(root=root, path=str(root / "src")),
    )
    assert result.status == "success"
    assert seen["paths"] == [str(kept)]
