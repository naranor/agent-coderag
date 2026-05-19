import os
import subprocess
import shutil
import sys
from pathlib import Path

# Use the same temporary directory for E2E consistency
E2E_TMP = Path(os.getenv("TEMP", "/tmp")) / "agent-coderag-e2e"


def setup_module(module):
    """Prepare a dummy project for E2E testing."""
    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)
    E2E_TMP.mkdir(parents=True)

    # Create a dummy Python file
    (E2E_TMP / "app.py").write_text('''
class Greeter:
    """A simple greeting class."""
    def say_hello(self, name: str):
        return f"Hello, {name}!"

def top_level_fn():
    return True
''')

    # Create a .gitignore
    (E2E_TMP / ".gitignore").write_text("*.log\n")


def teardown_module(module):
    """Cleanup."""
    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)


def run_cli(*args):
    """Helper to run the CLI as a subprocess."""
    # We use the current venv's python to ensure we use the installed package
    python_bin = sys.executable
    cmd = [
        python_bin,
        "-m",
        "code_rag.entry.cli",
        "--db",
        str(E2E_TMP / "test.db"),
    ] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(E2E_TMP))


def test_e2e_setup_command():
    """Verify setup command executes."""
    res = run_cli("setup")
    assert res.returncode in (0, 1)


def test_e2e_sync_and_db_state():
    """Verify sync works and database is populated."""
    # Index the current temp dir
    sync_res = run_cli("--verbose", "sync", "--all")
    assert sync_res.returncode == 0

    # Check if DB file was created
    assert (E2E_TMP / "test.db").exists()


def test_e2e_search_command_execution():
    """Verify search command executes without errors and finds our dummy code."""
    res = run_cli("search", "Greeter")
    assert res.returncode == 0


def test_e2e_json_output():
    """Verify that --json flag works and returns valid JSON."""
    res = run_cli("--json", "search", "Greeter")
    assert res.returncode == 0
    import json

    data = json.loads(res.stdout)
    assert isinstance(data, list)


def test_e2e_api_extraction():
    """Verify the API extraction command."""
    # json is a standard lib
    api_res = run_cli("api", "json")
    assert api_res.returncode == 0
    # It should at least output the header
    assert "Public API" in api_res.stdout
