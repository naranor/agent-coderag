import subprocess
import sys


def test_help_entry_does_not_import_heavy_stack():
    code = """
import sys
from code_rag.entry.args import main
assert "litellm" not in sys.modules
assert "duckdb" not in sys.modules
assert "onnxruntime" not in sys.modules
sys.argv = ["agent-coderag", "--help"]
main()
assert "litellm" not in sys.modules
assert "duckdb" not in sys.modules
assert "onnxruntime" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
