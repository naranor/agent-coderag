import asyncio
from pathlib import Path

from code_rag import CodeRAG


async def main():
    root = Path(__file__).resolve().parent
    db_path = root / "example_knowledge.db"

    async with CodeRAG(db=str(db_path), root=root) as rag:
        print(f"Indexing {__file__}...")
        await rag.sync(__file__)

        query = "How to setup the RAG components?"
        print(f"\nSearching for: '{query}'")
        results = await rag.search(query, limit=2)

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

    if db_path.exists():
        db_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
