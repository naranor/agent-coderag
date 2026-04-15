#!/usr/bin/env python3
"""Quick benchmark for token compression on agent-coderag codebase."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, '.')

from code_rag.entry import cli
from code_rag.core.manager import CodeRAGManager
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.parsers.ast_index import AstIndexParser
from code_rag.intelligence.embedder import Embedder
from code_rag.intelligence.distiller import Distiller, DistillerConfig


async def main():
    # Quick benchmark config: disable LLM distillation for speed
    config = DistillerConfig(provider="off")
    embedder = Embedder()
    storage = DuckDBStorage(".quick_bench.db", embedder=embedder)
    parser = AstIndexParser()
    distiller = Distiller(config)
    manager = CodeRAGManager(storage, parser, distiller)

    # Index only code_rag/ directory
    project = "code_rag"
    paths = [str(p) for p in Path(project).rglob("*.py") if cli.should_index(p)]
    print(f"Indexing {len(paths)} files in {project}/...")

    await manager.sync_project(paths, force_distill=True)
    print("Indexing complete.\n")

    # Test queries
    queries = [
        "extract API signatures",
        "delta synchronization",
        "embedding generation",
        "vector similarity search",
        "parse Python AST",
    ]

    total_summary_tokens = 0
    total_full_tokens = 0

    print("="*60)
    print("TOKEN COMPRESSION BENCHMARK")
    print("="*60)

    for query in queries:
        results = await manager.search(query, limit=5)
        summary_toks = sum(len(r.summary.split()) for r in results if r.summary)
        full_toks = sum(len(r.code.split()) for r in results if r.code)
        total_summary_tokens += summary_toks
        total_full_tokens += full_toks

        if results:
            compression = summary_toks / full_toks if full_toks else 1.0
            print(f"\nQuery: '{query}'")
            print(f"  Top result: {results[0].file}:{results[0].start_line}")
            print(f"  Summary tokens: {summary_toks}, Full tokens: {full_toks}")
            print(f"  Compression: {compression:.1%}")
        else:
            print(f"\nQuery: '{query}' — no results")

    print(f"\n{'='*60}")
    overall = total_summary_tokens / total_full_tokens if total_full_tokens else 1.0
    print(f"Overall token compression: {overall:.1%}")
    print(f"Summary tokens: {total_summary_tokens} | Full tokens: {total_full_tokens}")
    print(f"Token savings: {total_full_tokens - total_summary_tokens} tokens per query")
    print("="*60)

    # Precision estimate: we sample known matches
    # Count how many results actually contain query terms
    hits = 0
    total = 0
    for query in queries:
        results = await manager.search(query, limit=5)
        for r in results:
            total += 1
            if query.lower() in r.summary.lower() or query.lower() in r.code.lower():
                hits += 1
    if total > 0:
        print(f"\nHeuristic precision (query term in result): {hits/total:.1%}")

    # Save results
    with open("quick_benchmark.json", "w") as f:
        import json
        json.dump({
            "overall_compression": overall,
            "total_summary_tokens": total_summary_tokens,
            "total_full_tokens": total_full_tokens,
            "files_indexed": len(paths),
            "queries": len(queries),
        }, f, indent=2)
    print("\nResults saved to quick_benchmark.json")


if __name__ == "__main__":
    asyncio.run(main())
