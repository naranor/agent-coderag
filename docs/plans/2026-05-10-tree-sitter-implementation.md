# Tree-Sitter Universal Parser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transition to tree-sitter for multi-language parsing with minimal token footprint and high precision.

**Architecture:**
- `languages.py`: A central registry for 25+ languages containing node type mappings and stubbing rules.
- `tree_sitter.py`: A universal parser implementing dynamic grammar loading and "Hole-Punching" signature extraction.
- `multi_parser.py`: Updated to favor tree-sitter for supported extensions.

**Tech Stack:** Python, tree-sitter.

---

### Task 1: Create Language Registry

**Files:**
- Create: `code_rag/parsers/languages.py`
- Test: `tests/test_languages.py`

**Step 1: Define the mapping structure**
Implement `LanguageConfig` dataclass and a comprehensive `LANGUAGE_MAP`.

```python
from dataclasses import dataclass
from typing import List, Set

@dataclass
class LanguageConfig:
    package: str
    entities: Set[str]
    body_fields: List[str]
    fallback_bodies: Set[str]
    stub_suffix: str
    canonical_map: dict[str, str] # Maps raw type -> MODULE/CLASS/FUNCTION
```

**Step 2: Add mappings for 25 languages**
Populate the map based on the research (Python, JS, Java, C++, Rust, Go, etc.).

**Step 3: Commit**
```bash
git add code_rag/parsers/languages.py
git commit -m "feat(parser): add universal language registry"
```

---

### Task 2: Implement Tree-Sitter Parser Logic

**Files:**
- Create: `code_rag/parsers/tree_sitter.py`
- Test: `tests/test_tree_sitter_parser.py`

**Step 1: Implement Dynamic Loading**
Write a helper to import `tree_sitter_<lang>` modules on demand and raise `GrammarNotFoundError`.

**Step 2: Implement "Hole-Punching" Extraction**
Write logic to find the body node and extract the signature by cutting out the body's byte range.

```python
def extract_signature(node, config):
    body = find_body(node, config)
    if not body:
        return node.text.decode('utf-8')

    start = node.start_byte
    body_start = body.start_byte
    body_end = body.end_byte
    end = node.end_byte

    # Text before body + stub + text after body (e.g. closing brace)
    sig = source[start:body_start].decode('utf-8') + config.stub_suffix + source[body_end:end].decode('utf-8')
    return sig
```

**Step 3: Implement Recursive Distillation**
Walk the tree and extract `KnowledgeUnit`s for all matching entities.

**Step 4: Commit**
```bash
git add code_rag/parsers/tree_sitter.py
git commit -m "feat(parser): implement universal TreeSitterParser"
```

---

### Task 3: Integrate with MultiParser

**Files:**
- Modify: `code_rag/parsers/multi_parser.py`
- Test: `tests/test_multi_parser.py`

**Step 1: Update `MultiParser.__init__`**
Initialize `TreeSitterParser` and use it for any extension found in `LANGUAGE_MAP`.

**Step 2: Run verification**
Run tests for Python and Java (existing) and ensure they now pass via Tree-Sitter (or fallback).

**Step 3: Commit**
```bash
git add code_rag/parsers/multi_parser.py
git commit -m "refactor(parser): switch to TreeSitterParser in MultiParser"
```

---

### Task 4: Error Handling & UX

**Step 1: Improve the `GrammarNotFoundError` message**
Ensure it's clearly actionable for the CLI agent.

**Step 2: Add logging for skipped files**
Log clearly why a file couldn't be parsed (missing dependency vs syntax error).

**Step 3: Commit**
```bash
git commit -am "chore(parser): refine error handling for missing grammars"
```
