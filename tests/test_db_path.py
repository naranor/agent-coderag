from code_rag.paths import default_db_path, resolve_db_path


def test_default_prefers_cwd_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "code_rag.db"
    legacy.write_text("")
    assert default_db_path(tmp_path) == legacy


def test_default_prefers_root_legacy_when_cwd_missing(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    root = tmp_path / "root"
    cwd.mkdir()
    root.mkdir()
    monkeypatch.chdir(cwd)
    legacy = root / "code_rag.db"
    legacy.write_text("")
    assert default_db_path(root) == legacy


def test_default_new_name_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_db_path(tmp_path) == tmp_path / ".coderag.db"


def test_explicit_relative_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        resolve_db_path("custom.db", root=tmp_path / "other") == tmp_path / "custom.db"
    )


def test_explicit_absolute_unchanged(tmp_path):
    p = tmp_path / "x.db"
    assert resolve_db_path(p, root=tmp_path) == p
