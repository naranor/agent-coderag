from __future__ import annotations
from pathlib import Path


def default_db_path(root: Path | None = None) -> Path:
    root_path = Path(root) if root is not None else Path.cwd()
    cwd_legacy = Path.cwd() / "code_rag.db"
    if cwd_legacy.exists():
        return cwd_legacy
    root_legacy = root_path / "code_rag.db"
    if root_legacy.exists():
        return root_legacy
    return root_path / ".coderag.db"


def project_relative_posix(path: Path, root: Path) -> str:
    """Posix path of ``path`` relative to ``root``. Raises ValueError when outside ``root``."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_db_path(db: str | Path | None, *, root: Path | None = None) -> Path:
    if db is not None:
        path = Path(db)
        return path if path.is_absolute() else Path.cwd() / path
    return default_db_path(root)
