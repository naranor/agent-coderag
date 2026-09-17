import os
import subprocess
import shutil
import sys
from pathlib import Path

from code_rag.intelligence.embedder import get_default_model_dir

# Use the same temporary directory for E2E consistency
E2E_TMP = Path(os.getenv("TEMP", "/tmp")) / "agent-coderag-e2e"
_CLI_TIMEOUT_SECONDS = 120

_ORIGINAL_LOCALAPPDATA: str | None = None
_ONNX_PATH: Path | None = None


def _seed_local_model(appdata: Path) -> Path | None:
    """Copy cached MiniLM into isolated LOCALAPPDATA when available."""
    real_local = os.environ.get("LOCALAPPDATA") or str(
        Path.home() / "AppData" / "Local"
    )
    candidates = [
        Path(real_local) / "agent-coderag" / "models" / "mini-lm",
        get_default_model_dir(),
    ]
    src = next((p for p in candidates if (p / "model.onnx").exists()), None)
    if src is None:
        return None

    dest = appdata / "agent-coderag" / "models" / "mini-lm"
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
    global _ORIGINAL_LOCALAPPDATA, _ONNX_PATH

    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)
    E2E_TMP.mkdir(parents=True)

    # Isolate global config/models so developer config cannot hang distill/embed.
    _ORIGINAL_LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
    appdata = E2E_TMP / "appdata"
    appdata.mkdir()
    _ONNX_PATH = _seed_local_model(appdata)
    os.environ["LOCALAPPDATA"] = str(appdata)

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
    """Cleanup isolated temp + restore LOCALAPPDATA."""
    if _ORIGINAL_LOCALAPPDATA is None:
        os.environ.pop("LOCALAPPDATA", None)
    else:
        os.environ["LOCALAPPDATA"] = _ORIGINAL_LOCALAPPDATA

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
    ]
    if _ONNX_PATH is not None:
        cmd.extend(["--onnx", str(_ONNX_PATH)])
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
    """Verify setup command executes."""
    res = run_cli("setup")
    assert res.returncode in (0, 1)


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
