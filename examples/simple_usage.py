import asyncio
from code_rag.core.manager import CodeRAGManager
from code_rag.storage.duckdb_impl import DuckDBStorage
from code_rag.parsers.ast_index import AstIndexParser
from code_rag.intelligence.embedder import Embedder
from code_rag.intelligence.distiller import Distiller, DistillerConfig


async def main():
    # 1. Setup components
    # We use a temporary database for this example
    db_path = "example_knowledge.db"

    # Initialize embedder (will automatically find models in global cache)
    embedder = Embedder()
    storage = DuckDBStorage(db_path, embedder=embedder)
    parser = AstIndexParser()

    # Configure distiller (optional, but recommended for intent extraction)
    # If not configured, CodeRAG uses fallback name-based embeddings
    config = DistillerConfig.load()
    distiller = Distiller(config)

    # 2. Initialize Manager
    manager = CodeRAGManager(storage, parser, distiller)

    # 3. Index a file (for example, this script itself)
    print(f"Indexing {__file__}...")
    await manager.sync_file(__file__)

    # 4. Search for something in the code
    query = "How to initialize the manager?"
    print(f"\nSearching for: '{query}'")

    results = await manager.search(query, limit=2)

    for i, unit in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Name: {unit.name} ({unit.kind.value})")
        print(f"  Path: {unit.path}")
        if unit.summary:
            print(f"  Intent: {unit.summary}")

    # Cleanup (optional)
    # os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
