import os
import subprocess
import shutil
import time
import pytest
from pathlib import Path

# Paths for E2E tests
E2E_ROOT = Path(__file__).parent
E2E_TMP = E2E_ROOT / "tmp_workspace"
E2E_DB = E2E_ROOT / "e2e_test.db"
# Use default model path or from environment
ONNX_MODEL = os.getenv("RAG_ONNX_PATH", "models/bge-small-en-v1.5.onnx")

def run_cli(*args):
    """Executes code-rag CLI and returns output."""
    # Ensure environment is passed to preserve PATH for ast-index
    env = os.environ.copy()
    cmd = ["python3", "-m", "code_rag.entry.cli", "--db", str(E2E_DB)]
    if os.path.exists(ONNX_MODEL):
        cmd.extend(["--onnx", ONNX_MODEL])
    cmd.extend(args)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result

@pytest.fixture(scope="module", autouse=True)
def setup_workspace():
    """Sets up a temporary project structure with some python files."""
    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)
    if E2E_DB.exists():
        try:
            os.remove(E2E_DB)
        except: pass
        
    E2E_TMP.mkdir(parents=True)
    
    # 1. Create a math utility
    (E2E_TMP / "math_utils.py").write_text("""
def calculate_hypotenuse(a: float, b: float) -> float:
    \"\"\"Calculates the hypotenuse of a right triangle.\"\"\"
    import math
    return math.sqrt(a**2 + b**2)
""")

    # 2. Create a greet utility
    (E2E_TMP / "greet.py").write_text("""
class Greeter:
    def say_hello(self, name: str):
        \"\"\"Greets the person with a name.\"\"\"
        return f"Hello, {name}!"
""")

    yield
    
    if E2E_TMP.exists():
        shutil.rmtree(E2E_TMP)
    if E2E_DB.exists():
        try:
            os.remove(E2E_DB)
        except: pass

def test_e2e_sync_and_db_state():
    """Verify sync works and database is populated."""
    
    original_cwd = os.getcwd()
    os.chdir(E2E_TMP)
    try:
        sync_res = run_cli("sync", "--all")
        assert sync_res.returncode == 0
        assert "Indexing 2 files" in sync_res.stderr
        
        # Give some time for FS / locks to settle if needed
        time.sleep(1)

        import duckdb
        conn = duckdb.connect(str(E2E_DB))
        try:
            count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
            assert count >= 2
            
            names = [r[0] for r in conn.execute("SELECT name FROM units").fetchall()]
            # Check if at least file names or symbols are there
            assert any(n in names for n in ["math_utils.py", "greet.py", "Greeter", "calculate_hypotenuse"])
        finally:
            conn.close()

    finally:
        os.chdir(original_cwd)

def test_e2e_search_command_execution():
    """Verify search command executes without errors."""
    # Ensure DB is not locked by previous test
    time.sleep(1)
    res = run_cli("search", "anything")
    assert res.returncode == 0
    assert "Results" in res.stdout or "No results found" in res.stdout

def test_e2e_api_extraction():
    """Verify the API extraction command."""
    api_res = run_cli("api", "json")
    assert api_res.returncode == 0
    assert "Public API for 'json':" in api_res.stdout
    assert "JSONEncoder" in api_res.stdout
