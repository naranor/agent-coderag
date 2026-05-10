# Design Doc: Transition to Tree-Sitter for Universal Parsing

## Status
- **Date**: 2026-05-10
- **Author**: Gemini CLI
- **Status**: Approved

## Context
The current `agent-coderag` parsing system relies on a mix of `ast` (Python) and `javalang` (Java). Supporting additional languages requires writing custom parsers for each, which is not scalable. We need a universal parsing engine that can support 25+ languages with minimal overhead and high precision.

## Objectives
1.  Implement a universal parsing engine using `tree-sitter`.
2.  Enable support for 25+ popular programming languages.
3.  Minimize token usage by extracting only API signatures (classes, functions, methods).
4.  Provide "On-Demand" language support (dynamic installation).
5.  Maintain "AST Mirroring" (honoring original tree-sitter node types).

## Proposed Architecture

### 1. Universal Tree-Sitter Parser (`TreeSitterParser`)
-   **Dynamic Loading**: Grammars are loaded via `importlib`. If missing, an actionable error message is provided.
-   **Entity Discovery**: Uses a metadata-driven approach to identify definition nodes (functions, classes).
-   **Signature Extraction (Hole-Punching)**:
    -   API Signature = Full Node Text - Body Node Text.
    -   Body nodes are identified by fields (e.g., `body`) or fallbacks (e.g., `block`).
-   **Stub Injection**: Appends a language-specific suffix (`;`, `...`) to signify hidden implementation.

### 2. Language Configuration Module (`languages.py`)
-   Acts as a central registry for all supported languages.
-   Stores:
    -   Extension to package mapping.
    -   Target entity node types.
    -   Body field names and fallback types.
    -   Stub suffixes.

### 3. Error Handling
-   `GrammarNotFoundError`: Triggers an error message designed for AI/User intervention: `[MISSING DEPENDENCY] Please run 'pip install tree-sitter-<lang>'`.

## Data Model Changes
-   `KnowledgeUnit.kind` will now store the raw `tree-sitter-language` node type (e.g., `function_definition`, `impl_item`) to provide maximum context to AI agents.

## Implementation Steps
1.  Create `code_rag/parsers/languages.py` with initial mapping for 25 languages.
2.  Implement `code_rag/parsers/tree_sitter.py`.
3.  Update `code_rag/parsers/multi_parser.py` to route through the new universal parser.
4.  Add unit and integration tests.
