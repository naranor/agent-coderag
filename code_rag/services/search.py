from pathlib import Path

from code_rag.core.interfaces import IStorage
from code_rag.core.models import KnowledgeUnit


def format_unit_path(stored: str, *, root: Path, relative_paths: bool) -> str:
    """Format a stored index path for search output. Does not read or write storage."""
    if Path(stored).is_absolute():
        return stored
    if relative_paths:
        return stored
    return str((root / stored).resolve())


async def run_search(
    storage: IStorage, query: str, *, limit: int = 5
) -> list[KnowledgeUnit]:
    """Performs semantic search across indexed units."""
    return await storage.search_units(query, limit=limit)
