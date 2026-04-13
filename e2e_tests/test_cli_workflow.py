import os
import subprocess
import pytest
import shutil
import time
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
E2E_TMP = PROJECT_ROOT / "e2e_tests" / "tmp_workspace"
DB_PATH = PROJECT_ROOT / "e2e_tests" / "e2e_test.db"

def run_cli(*args):
    """Helper to run the CLI tool."""
    cmd = [
        "python3", "-m", "code_rag.entry.cli",
        "--db", str(DB_PATH),
    ] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True)

@pytest.fixture(scope="module", autouse=True)
def setup_e2e_env():
    """Sets up a temporary environment for E2E tests."""
    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)
    E2E_TMP.mkdir(parents=True)
    
    # Create some dummy python files
    (E2E_TMP / "greet.py").write_text("""
class Greeter:
    def say_hello(self, name: str):
        return f"Hello {name}"
""")
    
    (E2E_TMP / "math_utils.py").write_text("""
def calculate_hypotenuse(a, b):
    \"\"\"Calculates hypotenuse using Pythagorean theorem.\"\"\"
    return (a**2 + b**2)**0.5
""")

    # Ensure fresh DB
    if DB_PATH.exists():
        DB_PATH.unlink()
        
    yield
    
    # Cleanup
    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)
    if DB_PATH.exists():
        DB_PATH.unlink()

def test_e2e_setup_command():
    """Verify setup command downloads (or finds) models."""
    res = run_cli("setup")
    assert res.returncode == 0
    assert "Setup complete" in res.stdout

def test_e2e_sync_and_db_state():
    """Verify sync works and database is populated."""
    original_cwd = os.getcwd()
    os.chdir(E2E_TMP)
    try:
        # Use --verbose to check for the indexing message
        sync_res = run_cli("--verbose", "sync", "--all")
        assert sync_res.returncode == 0
        assert "Indexing 2 files" in sync_res.stderr
        
        # Check if database file was created
        assert DB_PATH.exists()
    finally:
        os.chdir(original_cwd)

def test_e2e_search_command_execution():
    """Verify search command executes without errors and finds our dummy code."""
    # Ensure DB is ready
    time.sleep(1)
    
    # Search for something that should exist
    res = run_cli("search", "Greeter")
    assert res.returncode == 0
    # New format: [class] Greeter | greet.py
    assert "[class] Greeter" in res.stdout
    
    # Search for math logic
    res = run_cli("search", "pythagorean theorem")
    assert res.returncode == 0
    assert "calculate_hypotenuse" in res.stdout

def test_e2e_json_output():
    """Verify that --json flag works and returns valid JSON."""
    res = run_cli("--json", "search", "Greeter")
    assert res.returncode == 0
    import json
    data = json.loads(res.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    names = [unit["name"] for unit in data]
    assert "Greeter" in names
