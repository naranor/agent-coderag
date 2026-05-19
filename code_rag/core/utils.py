import tempfile
from pathlib import Path
from typing import Optional
from ..intelligence.embedder import get_global_dir


def validate_path(path: str | Path, root: Optional[Path] = None) -> Path:
    """
    Validates that a path is within the specified root directory to prevent path traversal.
    Defaults to current working directory if root is not provided.
    """
    if root is None:
        # For CLI tools, we usually allow paths within the current project
        # and standard system locations.
        root = Path.cwd().resolve()
    else:
        root = root.resolve()

    resolved_path = Path(path).resolve()

    # Special case: handle non-existent paths for new DBs/files
    # We check the parent directory in that case
    if not resolved_path.exists():
        check_path = resolved_path.parent.resolve()
    else:
        check_path = resolved_path

    # Check if within root
    if str(check_path).startswith(str(root)):
        return resolved_path

    # Exception: allow system temp dir (important for tests)
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if str(check_path).startswith(str(temp_dir)):
        return resolved_path

    # Exception: allow reading from global agent-coderag data dir
    global_dir_resolved = get_global_dir().resolve()
    if str(check_path).startswith(str(global_dir_resolved)):
        return resolved_path

    # Exception: allow standard development caches in user home
    home = Path.home().resolve()
    dev_caches = [
        home / ".m2",
        home / ".gradle",
        home / ".nuget",
        home / ".cargo",
    ]
    for cache in dev_caches:
        if str(check_path).startswith(str(cache)):
            return resolved_path

    # Exception: if we are in a git repo, allow anything within that repo
    # This helps when os.chdir() is used in sub-tasks
    repo_root = find_directory_upwards(Path.cwd(), ".git")
    if repo_root and str(check_path).startswith(str(repo_root.parent.resolve())):
        return resolved_path

    raise ValueError(
        f"Security Risk: Path '{path}' is outside permitted directory '{root}'"
    )


def find_directory_upwards(
    start_path: Path, target_name: str, max_levels: int = 5
) -> Optional[Path]:
    """
    Searches for a directory/file upwards from start_path.
    Consolidates duplicated logic from multiple providers.
    """
    curr = start_path.resolve()
    for _ in range(max_levels):
        target = curr / target_name
        if target.exists():
            return target
        if curr.parent == curr:
            break
        curr = curr.parent
    return None
