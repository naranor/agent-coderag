from code_rag.parsers.languages import (
    LANGUAGE_MAP,
    EXTENSION_TO_LANGUAGE,
    LanguageConfig,
)


def test_language_map_contains_expected_languages():
    expected_languages = [
        "python",
        "javascript",
        "typescript",
        "java",
        "cpp",
        "c_sharp",
        "c",
        "go",
        "rust",
        "php",
        "ruby",
        "swift",
        "kotlin",
        "sql",
        "bash",
        "r",
        "dart",
        "scala",
        "lua",
        "perl",
        "haskell",
        "objective_c",
        "zig",
        "elixir",
        "julia",
    ]
    for lang in expected_languages:
        assert lang in LANGUAGE_MAP
        assert isinstance(LANGUAGE_MAP[lang], LanguageConfig)


def test_language_config_structure():
    config = LANGUAGE_MAP["python"]
    assert config.package == "tree_sitter_python"
    assert "class_definition" in config.entities
    assert "function_definition" in config.entities
    assert "body" in config.body_fields
    assert "block" in config.fallback_bodies
    assert config.stub_suffix == " ..."
    assert config.canonical_map["class_definition"] == "CLASS"
    assert config.canonical_map["function_definition"] == "FUNCTION"


def test_extension_mapping():
    assert EXTENSION_TO_LANGUAGE[".py"] == "python"
    assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
    assert EXTENSION_TO_LANGUAGE[".java"] == "java"
    assert EXTENSION_TO_LANGUAGE[".rs"] == "rust"
    assert EXTENSION_TO_LANGUAGE[".go"] == "go"


def test_all_languages_in_extension_map_are_in_language_map():
    for lang in EXTENSION_TO_LANGUAGE.values():
        assert lang in LANGUAGE_MAP
