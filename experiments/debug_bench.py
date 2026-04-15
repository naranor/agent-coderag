#!/usr/bin/env python3
"""Debug: test individual components."""
import sys
sys.path.insert(0, '.')

from code_rag.intelligence.embedder import Embedder
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.parsers.ast_index import AstIndexParser
from pathlib import Path

print("1. Loading embedder...")
emb = Embedder()
print("   OK")

print("2. Creating storage...")
storage = DuckDBStorage(".debug.db", embedder=emb)
print("   OK")

print("3. Creating parser...")
parser = AstIndexParser()
print("   OK")

print("4. Parsing one file...")
test_file = "code_rag/entry/cli.py"
import asyncio
units = asyncio.run(parser.distill_file(test_file))
print(f"   Parsed {len(units)} units")

print("5. Computing embeddings...")
for u in units[:2]:
    print(f"   Unit: {u.name}, generating embedding...")
    embu = emb.embed(u.summary)
    print(f"   Embedding shape: {embu.shape}")

print("\nAll components work! Benchmark should work.")
