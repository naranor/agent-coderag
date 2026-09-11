from pathlib import Path


def test_readme_remote_embeddings_not_fully_offline():
    content = Path("README.md").read_text(encoding="utf-8")
    assert "### Offline Mode (No Provider)" in content
    assert "--embedding-url" in content
    assert "--embedding-model" in content
    lowered = content.lower()
    assert "not" in lowered and "100% offline" in lowered
    assert "rebuild" in lowered
