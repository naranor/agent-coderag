#!/usr/bin/env python3
"""Synchronous benchmark to avoid async issues."""
import sys
from pathlib import Path

sys.path.insert(0, '.')

from code_rag.parsers.ast_index import AstIndexParser
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.intelligence.embedder import Embedder

def main():
    print("SYNC BENCHMARK TEST\n")

    # 1. Parser
    print("1. Loading parser...")
    parser = AstIndexParser()
    print("   OK")

    # 2. Embedder
    print("2. Loading embedder...")
    embedder = Embedder()
    print("   OK")

    # 3. Storage (sync init)
    print("3. Creating storage...")
    storage = DuckDBStorage(".sync_bench.db", embedder=embedder, create=True)
    print("   OK")

    # 4. Parse single file (sync)
    print("4. Parsing sample file...")
    test_file = "code_rag/entry/cli.py"
    print(f"   File: {test_file}")
    print(f"   Exists: {Path(test_file).exists()}")
    # Get text
    text = Path(test_file).read_text()
    print(f"   File size: {len(text)} chars")

    # Use distill_file (async wrapper from parser)
    import asyncio
    units = asyncio.run(parser.distill_file(test_file))
    print(f"   Parsed {len(units)} knowledge units")
    for u in units[:3]:
        print(f"     • {u.kind}: {u.name} ({u.file}:{u.start_line})")
        print(f"       summary preview: {u.summary and u.summary[:60]}")

    print("\n✓ Baseline works — embedder + storage + parser OK")
    print("  Ready to add distiller + retrieval for full benchmark.")

if __name__ == "__main__":
    main()
