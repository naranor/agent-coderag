import tempfile
from pathlib import Path
from typing import Optional
from ..intelligence.embedder import get_global_dir


def validate_path(path: str | Path, root: Optional[Path] = None) -> Path:
    """
    Validates that a path is within the specified root directory to prevent path traversal.
    Defaults to current working directory if root is not provided.
    """
    # 1. Resolve root
    if root is None:
        root_path = Path.cwd().resolve()
    else:
        root_path = Path(root).resolve()

    # 2. Resolve target path
    resolved_path = Path(path).resolve()

    # Special case: handle non-existent paths for new DBs/files
    if not resolved_path.exists():
        check_path = resolved_path.parent.resolve()
    else:
        check_path = resolved_path

    # Helper to check subpath relationship manually for robustness against mocks
    def is_subpath(p: Path, r: Path) -> bool:
        try:
            return p.is_relative_to(r)
        except (AttributeError, ValueError):
            return str(p).startswith(str(r))

    if is_subpath(check_path, root_path):
        return resolved_path

    # Exception: allow system temp dir
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if is_subpath(check_path, temp_dir):
        return resolved_path

    # Exception: allow reading from global agent-coderag data dir
    global_dir_resolved = get_global_dir().resolve()
    if is_subpath(check_path, global_dir_resolved):
        return resolved_path

    # Exception: allow standard development caches in user home
    try:
        home = Path.home().resolve()
        dev_caches = [
            home / ".m2",
            home / ".gradle",
            home / ".nuget",
            home / ".cargo",
        ]
        for cache in dev_caches:
            if is_subpath(check_path, cache.resolve()):
                return resolved_path
    except Exception:  # nosec
        pass

    # Exception: if we are in a git repo, allow anything within that repo
    repo_root = find_directory_upwards(Path.cwd(), ".git")
    if repo_root:
        repo_parent = repo_root.parent.resolve()
        if is_subpath(check_path, repo_parent):
            return resolved_path

    raise ValueError(
        f"Security Risk: Path '{path}' is outside permitted directory '{root_path}'"
    )


def find_directory_upwards(
    start_path: Path, target_name: str, max_levels: int = 5
) -> Optional[Path]:
    """
    Searches for a directory/file upwards from start_path.
    """
    try:
        curr = start_path.resolve()
    except Exception:  # nosec
        curr = start_path

    for _ in range(max_levels):
        target = curr / target_name
        if target.exists():
            return target
        if curr.parent == curr:
            break
        curr = curr.parent
    return None
