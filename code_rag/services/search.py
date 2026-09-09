from code_rag.core.models import KnowledgeUnit


async def run_search(manager, query: str, *, limit: int = 5) -> list[KnowledgeUnit]:
    """Performs semantic search across indexed units."""
    return await manager.search(query, limit=limit)
