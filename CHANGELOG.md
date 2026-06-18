## [1.3.1] - 2026-06-18

### Security
- **Critical:** Fixed Arbitrary Code Execution (ACE) vulnerability ([GHSA-wg5p-8h9p-3mr7](https://github.com/naranor/agent-coderag/security/advisories/GHSA-wg5p-8h9p-3mr7)) triggered during Gradle dependency discovery. `agent-coderag sync` now strictly enforces the use of the system-installed `gradle` binary and ignores local wrapper scripts to prevent executing untrusted repository code.


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-05-19

### Added
- **Global Docstring Extraction:** Parser now extracts and stores code comments/docstrings for all supported languages.
- **Recursive API Discovery:** Improved JS/TS and Rust providers to follow local exports and modules (up to 3 levels deep).
- **Gradle Support:** Added automated dependency resolution and JAR path caching for Gradle-based Java projects.
- **Garbage Collection:** Automatic removal of stale records from the database when files are deleted or modified.
- **Worker Pool:** Memory-efficient indexing using an asynchronous task queue to handle large projects.
- **Safety & Validation:** Implemented Path Traversal protection and strict sanitization for all subprocess arguments.
- **Resource Management:** Added explicit `close()` methods for Storage and Embedder to ensure clean session termination.

### Changed
- **Performance:** Fixed N+1 query problem by batch-fetching all unit relations in a single database request.
- **Robustness:** Replaced silent "zero-vector" failures in the embedder with explicit `IntelligenceError` exceptions.
- **Stability:** Added timeouts (30-120s) to all external subprocess calls (Maven, Gradle, Cargo, Go).
- **Architecture:** Centralized project-wide constants and unified error handling with a new typed exception hierarchy.

### Fixed
- **SemVer Sorting:** Corrected lexicographical version sorting for Java and C# dependencies (now 1.10.0 > 1.9.0).
- **Decoding:** Added fault-tolerant UTF-8 decoding in the parser to prevent crashes on non-standard source files.
- **Anonymous Blocks:** Improved naming for anonymous functions and lambda blocks by resolving them from parent context.

## [1.2.2] - 2026-05-09

### Added
- Upgraded repository to professional Open Source standards (Elite OS).
- Added `CODE_OF_CONDUCT.md` and `PULL_REQUEST_TEMPLATE.md`.
- Integrated `CodeQL` automated security scanning (standard setup).
- Added `Makefile` for common tasks (install, test, lint).
- Added `.editorconfig` for consistent coding style.

### Changed
- Refactored `README.md` into a high-conversion landing page structure.
- Removed Unicode emojis from documentation for a cleaner professional look.
- Restored architecture Mermaid diagrams.

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
