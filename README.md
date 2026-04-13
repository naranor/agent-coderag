# CodeRAG

Semantic search and code distillation utility for Gemini CLI.

## Features
- Semantic search using ONNX embeddings.
- Code distillation for compact context.
- DuckDB storage for indexed units.
- AST-based parsing for precise symbol extraction.

## Installation

### From Source
To install the package into your system or virtual environment:
```bash
# Standard installation
pip install .

# Editable mode (for development)
pip install -e .
```

### Dependencies
Ensure you have the required libraries:
```bash
pip install -r requirements.txt
```

## Usage

After installation, the `code-rag` command will be available in your terminal.

### Indexing
Sync your project files into the knowledge base:
```bash
# Sync all python files in current directory
code-rag sync --all

# Sync a specific file or directory
code-rag sync path/to/code/
```

### Semantic Search
Search your codebase using natural language:
```bash
code-rag search "how does the embedding logic work?" --limit 3
```

### API Discovery
Extract the public API of any installed library:
```bash
code-rag api pydantic_ai
```

## Testing
To run unit and integration tests:
```bash
pytest
```

To run E2E tests (not included in default pytest run):
```bash
pytest e2e_tests/
```
