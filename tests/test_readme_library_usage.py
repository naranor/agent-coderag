from pathlib import Path


def test_readme_mentions_library_usage():
    content = Path("README.md").read_text(encoding="utf-8")
    assert "## Library Usage" in content
    assert "from code_rag import CodeRAG" in content
    assert "await rag.sync(index_all=True)" in content
    assert "default_db_path" in content
