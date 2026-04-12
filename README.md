# CodeRAG

Semantic search and code distillation utility for Gemini CLI.

## Features
- Semantic search using ONNX embeddings.
- Code distillation for compact context.
- DuckDB storage for indexed units.
- AST-based parsing for precise symbol extraction.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python3 rag.py sync
python3 rag.py search "how does it work?"
```
