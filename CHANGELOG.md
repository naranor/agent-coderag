## [Unreleased]

### Added
- Index paths are stored relative to the project root. The first `sync` rewrites an absolute index in place when file hashes match, without re-distilling unchanged units. `relative_paths` (config, `CodeRAG(relative_paths=...)`, and `search --relative-paths`) returns those relative paths; the default search path is absolute under the current root.

### Fixed
- Gitignore patterns are matched against the path relative to the project root, so a worktree checked out under a gitignored `.worktrees/` directory is indexed.

## [1.4.1] - 2026-09-21

### Added
- Public `ErrorCode.EMBEDDINGS_MISSING` when a read-only open finds base tables but no `unit_embeddings` table.
- CLI `--json` error objects may include `"code"` when the exception is a `CodeRAGError` with a code.

### Fixed
- Search no longer surfaces a raw DuckDB catalog error when the embeddings table was never created; agents get an actionable `EMBEDDINGS_MISSING` instead.
- `agent-coderag --help` no longer imports LiteLLM, DuckDB, or ONNX. LiteLLM loads only when distillation or remote embeddings actually run.

## [1.4.0] - 2026-09-18

### Breaking
- **Pin before upgrading:** 1.4.0 changes library storage semantics. Use `agent-coderag<1.4` until you migrate callers and scripts.
- Removed `CodeRAGManager` and `code_rag.core.manager`; use-cases live in `code_rag.services.indexing`, `dependencies`, `search`, and `sync`.
- Removed `create_manager` / `build_manager` from `code_rag.services.factory` (use `create_stack` or the public `CodeRAG` facade).
- Removed `DuckDBStorage.open`; use `open_db_connection` from `code_rag.storage.db_connection` or the public `CodeRAG` facade.
- **Default `db` is unset:** `CodeRAG()` and CLI omit `--db` by default (`None`), not a hard-coded `code_rag.db`. Resolution order: cwd `code_rag.db` if exists → `{root}/code_rag.db` if exists → `{root}/.coderag.db` (create on RW ops). Use `default_db_path(root)` to preview.
- **New create default:** fresh projects get `{root}/.coderag.db` unless a legacy `code_rag.db` is found. Relative explicit `db=` resolves against process cwd, not `root`.
- **Ephemeral DuckDB per op:** the index file is not held open between `search`/`sync`/`api` calls; embedder/parser/distiller stay warm until `close()`.
- **Read-only search:** `search` opens read-only and does not create an empty database when the file is missing.
- **`api()` lazy storage:** API discovery no longer implies a warm DB/embedder; DuckDB opens read-only only when a provider needs it (e.g. Java JAR cache).
- **Serialized in-process ops:** overlapping tasks on one `CodeRAG` instance are queued (no parallel DB access in-process).
- **Stable storage errors:** file-lock timeouts surface as `StorageBusyError` with `ErrorCode.STORAGE_BUSY` instead of raw `duckdb` exceptions. `CodeRAGError.code` uses public `ErrorCode` values (`STORAGE_CORRUPT`, `EMBEDDING_MISMATCH`, …).

### Changed
- Added `connect_timeout_seconds` (default `5`; CLI `--connect-timeout`). `0` disables busy-wait retries.
- Exported `default_db_path`, `ErrorCode`, and `StorageBusyError` from the public package.
- Documented DuckDB WAL sidecars beside the index file.

### Fixed
- Cross-process lock contention now retries until timeout, then fails with `STORAGE_BUSY`.
- Corrupt metadata and embedding-dimension mismatches map to `STORAGE_CORRUPT` / `EMBEDDING_MISMATCH`.
- Distiller no longer uses implicit LLM defaults (`model=auto`, `provider=openai`, localhost API). Without explicit `model` + `api_base` + `provider`, `summarize` skips LiteLLM (true offline distillation fallback).
- CLI e2e isolates `LOCALAPPDATA` and `XDG_CACHE_HOME`, seeds MiniLM into the temp cache (or `CODERAG_E2E_ONNX`), and `pytest.skip`s with a clear message when no ONNX is available.
- `CodeRAG.config()` invalidates the process embedder only when embedding fields change (`--clear-embedding` / embedding-*); distill-only updates refresh Distiller and keep the embedder.

## [1.3.5] - 2026-09-12

### Fixed
- Remote embeddings accept LiteLLM `aembedding` items as dicts (llama.cpp / OpenAI-compat servers).

## [1.3.4] - 2026-09-11

### Added
- OpenAI-compatible remote embeddings (LiteLLM `aembedding`) alongside local ONNX MiniLM.
- CLI flags `--embedding-url`, `--embedding-key`, `--embedding-model`, `--embedding-provider`, and `--clear-embedding`.
- Per-file `sync` / `rebuild` error reporting: JSON `errors` list, human stderr summary, exit code 1.

### Changed
- Embedding dimension is autodetected and stored in DuckDB `index_meta`; `rebuild` drops embedding rows and re-binds.
- Embedder bind/probe is deferred until the first vector operation so `api` works without ONNX setup or a reachable embedding endpoint.
- `sync --all` re-embeds existing units even when the walk finds no indexable files.

### Fixed
- Concurrent `sync --all` workers no longer race the DuckDB connection (`KeyError: 'kind'`).
- File-level worker failures no longer report overall `success` after a partial index.

## [1.3.3] - 2026-09-09

### Added
- Public async library API: `from code_rag import CodeRAG` with typed results for sync, search, API discovery, config, setup, and rebuild.
- Shared service layer under `code_rag.services` so CLI and library share the same business logic.

### Changed
- CLI is now a thin adapter over the shared services (behavior and JSON output preserved).
- Added `requests` as a declared runtime dependency for setup HTTP checks.

### Documentation
- Documented library usage in README with an async example.

## [1.3.2] - 2026-08-25

### Security
- Made Maven and Gradle dependency execution opt-in through `--allow-build-execution`, preventing automatic execution of repository-controlled build logic.

### Fixed
- Avoided dependency-sync warnings for projects without Maven or Gradle build files.
- Pinned LiteLLM to `1.96.2` to preserve Python 3.10 compatibility.

### Documentation
- Documented the trusted-project build execution opt-in and updated the bundled CodeRAG intelligence skill.

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
