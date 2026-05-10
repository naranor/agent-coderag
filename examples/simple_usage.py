import asyncio
import os
from code_rag.core.manager import CodeRAGManager
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.parsers.multi_parser import MultiParser
from code_rag.intelligence.embedder import Embedder
from code_rag.intelligence.distiller import Distiller, DistillerConfig


async def main():
    # 1. Setup components
    # We use a temporary database for this example
    db_path = "example_knowledge.db"

    # Initialize embedder (will automatically find models in global cache)
    embedder = Embedder()
    storage = DuckDBStorage(db_path, embedder=embedder)

    # Use MultiParser which now uses Tree-Sitter for 25+ languages
    # It automatically detects the language and loads the grammar
    parser = MultiParser()

    # Configure distiller (optional, but recommended for intent extraction)
    # If not configured, CodeRAG uses fallback name-based metadata
    config = DistillerConfig.load()
    distiller = Distiller(config)

    # 2. Initialize Manager
    manager = CodeRAGManager(storage, parser, distiller)

    # 3. Index a file (this script itself)
    # Tree-Sitter will automatically detect this as a Python file
    print(f"Indexing {__file__}...")
    await manager.sync_file(__file__)

    # 4. Search for something in the code using semantic intent
    query = "How to setup the RAG components?"
    print(f"\nSearching for: '{query}'")

    results = await manager.search(query, limit=2)

    if not results:
        print(
            "No results found. (Make sure you have an LLM provider or at least signatures indexed)"
        )

    for i, unit in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Name: {unit.name} ({unit.kind.value})")
        print(f"  ID:   {unit.id}")
        print(f"  Path: {unit.path}")
        if unit.summary:
            print(f"  Intent: {unit.summary}")

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
