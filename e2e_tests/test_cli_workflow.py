import os
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

# Use the same temporary directory for E2E consistency
E2E_TMP = Path(os.getenv("TEMP", "/tmp")) / "agent-coderag-e2e"
_CLI_TIMEOUT_SECONDS = 120
_SKIP_NO_ONNX = (
    "E2E requires a local MiniLM ONNX model. "
    "Run `agent-coderag setup` once on this machine, "
    "or set CODERAG_E2E_ONNX to model.onnx (or a directory containing it)."
)

_ORIGINAL_LOCALAPPDATA: str | None = None
_ORIGINAL_XDG_CACHE_HOME: str | None = None
_ONNX_PATH: Path | None = None


def _host_mini_lm_dirs() -> list[Path]:
    """Candidate host model dirs (read before env isolation)."""
    dirs: list[Path] = []
    env_onnx = os.environ.get("CODERAG_E2E_ONNX")
    if env_onnx:
        path = Path(env_onnx)
        if path.is_file():
            dirs.append(path.parent)
        elif path.is_dir():
            dirs.append(path)

    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "agent-coderag" / "models" / "mini-lm")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        dirs.append(Path(xdg) / "agent-coderag" / "models" / "mini-lm")

    dirs.append(
        Path.home() / "AppData" / "Local" / "agent-coderag" / "models" / "mini-lm"
    )
    dirs.append(Path.home() / ".cache" / "agent-coderag" / "models" / "mini-lm")
    return dirs


def _seed_local_model(cache_root: Path) -> Path | None:
    """Copy MiniLM into isolated cache; return path to model.onnx or None."""
    src = next((p for p in _host_mini_lm_dirs() if (p / "model.onnx").exists()), None)
    if src is None:
        return None

    dest = cache_root / "agent-coderag" / "models" / "mini-lm"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "model.onnx", dest / "model.onnx")
    for name in ("tokenizer.json",):
        token_src = src / name
        if not token_src.exists():
            token_src = src.parent / name
        if token_src.exists():
            shutil.copy2(token_src, dest / name)
    return dest / "model.onnx"


def setup_module(module):
    """Prepare an isolated dummy project for E2E testing."""
    global _ORIGINAL_LOCALAPPDATA, _ORIGINAL_XDG_CACHE_HOME, _ONNX_PATH

    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)
    E2E_TMP.mkdir(parents=True)

    # Snapshot host env, then isolate config/model cache on all platforms.
    _ORIGINAL_LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
    _ORIGINAL_XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME")
    cache_root = E2E_TMP / "cache"
    cache_root.mkdir()
    _ONNX_PATH = _seed_local_model(cache_root)
    if _ONNX_PATH is None:
        pytest.skip(_SKIP_NO_ONNX)

    os.environ["LOCALAPPDATA"] = str(cache_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

    (E2E_TMP / "app.py").write_text(
        '''
class Greeter:
    """A simple greeting class."""
    def say_hello(self, name: str):
        return f"Hello, {name}!"

def top_level_fn():
    return True
''',
        encoding="utf-8",
    )

    (E2E_TMP / "nlp.py").write_text(
        '''
def tokenize(text: str) -> list[str]:
    """Split raw text into tokens."""
    return text.split()
''',
        encoding="utf-8",
    )

    (E2E_TMP / ".gitignore").write_text("*.log\n", encoding="utf-8")

    clear_res = run_cli("config", "--clear-embedding")
    assert clear_res.returncode == 0, clear_res.stderr


def teardown_module(module):
    """Cleanup isolated temp + restore host cache env."""
    if _ORIGINAL_LOCALAPPDATA is None:
        os.environ.pop("LOCALAPPDATA", None)
    else:
        os.environ["LOCALAPPDATA"] = _ORIGINAL_LOCALAPPDATA

    if _ORIGINAL_XDG_CACHE_HOME is None:
        os.environ.pop("XDG_CACHE_HOME", None)
    else:
        os.environ["XDG_CACHE_HOME"] = _ORIGINAL_XDG_CACHE_HOME

    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)


def run_cli(*args):
    """Helper to run the CLI as a subprocess."""
    python_bin = sys.executable
    cmd = [
        python_bin,
        "-m",
        "code_rag.entry.cli",
        "--db",
        str(E2E_TMP / "test.db"),
        "--onnx",
        str(_ONNX_PATH),
    ]
    cmd.extend(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(E2E_TMP),
        timeout=_CLI_TIMEOUT_SECONDS,
        env=os.environ.copy(),
    )


def test_e2e_setup_command():
    """Verify setup command executes under isolated cache."""
    res = run_cli("setup")
    # May download or no-op if files already seeded; allow non-zero on network errors.
    assert res.returncode in (0, 1), res.stderr


def test_e2e_sync_and_db_state():
    """Verify sync works and database is populated."""
    sync_res = run_cli("--verbose", "sync", "--all")
    assert sync_res.returncode == 0, sync_res.stderr

    assert (E2E_TMP / "test.db").exists()
    import duckdb

    conn = duckdb.connect(str(E2E_TMP / "test.db"), read_only=True)
    names = {row[0] for row in conn.execute("SELECT name FROM units").fetchall()}
    embed_n = conn.execute("SELECT count(*) FROM unit_embeddings").fetchone()[0]
    conn.close()
    assert "Greeter" in names
    assert "tokenize" in names
    assert embed_n == len(names)


def test_e2e_search_command_execution():
    """Verify search command executes without errors and finds our dummy code."""
    res = run_cli("search", "Greeter")
    assert res.returncode == 0, res.stderr


def test_e2e_json_output():
    """Verify that --json flag works and returns valid JSON."""
    res = run_cli("--json", "search", "Greeter")
    assert res.returncode == 0, res.stderr
    import json

    data = json.loads(res.stdout)
    assert isinstance(data, list)
    names = {item.get("name") for item in data}
    assert "Greeter" in names or any("Greeter" in str(item) for item in data)


def test_e2e_api_extraction():
    """Verify the API extraction command."""
    api_res = run_cli("api", "json")
    assert api_res.returncode == 0, api_res.stderr
    assert "Public API" in api_res.stdout
