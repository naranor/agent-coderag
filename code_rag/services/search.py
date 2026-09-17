from code_rag.core.interfaces import IStorage
from code_rag.core.models import KnowledgeUnit


async def run_search(
    storage: IStorage, query: str, *, limit: int = 5
) -> list[KnowledgeUnit]:
    """Performs semantic search across indexed units."""
    return await storage.search_units(query, limit=limit)
