# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-05-09

### Fixed
- Robust `.gitignore` handling: correctly ignores nested directories (like `venv/`) using the `pathspec` library.
- Cleanup: removed unused imports in CLI to satisfy linting.

## [1.2.0] - 2026-04-30

### Added
- Full Java support: indexing classes, methods, and signatures using `javalang`.
- Java API Discovery: automatically extracts public API from Maven (`~/.m2`) and Gradle (`~/.gradle`) caches using `javap`.
- Extensible `MultiParser` architecture to support multiple programming languages.
- Relation mapping: tracks imports and dependencies between modules.
- `.gitignore` support: project indexing now respects local ignore rules.
- Concurrency control: added `asyncio.Semaphore` to limit concurrent LLM requests.

### Fixed
- Correct classification of class methods in AST parser.
- Use of qualified names (QNames) for unit IDs to prevent collisions in nested structures.
- Robust database mapping in DuckDB: replaced fragile `SELECT *` with explicit column selection.
- Windows path handling: fixed issues with backslashes in indexing and tests.
- CI/CD stabilization: resolved various linting issues (Prospector, Bandit, Vulture).

## [0.1.0] - 2026-04-13

### Added
- Semantic search using ONNX embeddings.
- Code distillation for compact context.
- DuckDB storage for indexed units.
- AST-based parsing for precise symbol extraction.
- CLI with `sync`, `search`, and `api` commands.
