from pathlib import Path


def test_readme_documents_lifetime_and_db_resolve():
    content = Path("README.md").read_text(encoding="utf-8")
    assert "### Database & lifetime" in content
    assert "default_db_path" in content
    assert ".coderag.db" in content
    assert "code_rag.db" in content
    assert "connect_timeout_seconds" in content
    assert "STORAGE_BUSY" in content
    assert "read-only" in content.lower()
    assert "agent-coderag<1.4" in content


def test_readme_library_example_uses_default_db():
    content = Path("README.md").read_text(encoding="utf-8")
    assert "CodeRAG(root=root)" in content
    assert 'db="code_rag.db"' not in content


def test_security_documents_db_resolve_chain():
    content = Path("SECURITY.md").read_text(encoding="utf-8")
    assert ".coderag.db" in content
    assert "code_rag.db" in content
    assert ".code_rag.db" not in content


def test_changelog_unreleased_documents_14_breaking():
    content = Path("CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = content.split("## [Unreleased]")[1].split("## [1.3.5]")[0]
    assert "### Breaking" in unreleased
    assert "default_db_path" in unreleased
    assert "STORAGE_BUSY" in unreleased
    assert "agent-coderag<1.4" in unreleased
