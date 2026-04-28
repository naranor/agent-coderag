import logging
import asyncio
from typing import List, Optional
from .interfaces import IStorage, IParser, IIntelligence
from .models import KnowledgeUnit

logger = logging.getLogger(__name__)

class CodeRAGManager:
    """
    Orchestrates the RAG workflow: parsing, distillation, and storage.
    """
    
    def __init__(self, storage: IStorage, parser: IParser, intelligence: IIntelligence, max_concurrency: int = 10):
        self.storage = storage
        self.parser = parser
        self.intelligence = intelligence
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def sync_file(self, file_path: str, force_distill: bool = False):
        """
        Processes a single file and syncs it with the storage.
        """
        # 1. Parse AST to get units
        current_units = await self.parser.distill_file(file_path)
        
        for unit in current_units:
            # v5.40: Delta-distillation logic
            raw_code = unit.metadata.pop("raw_code", "")
            
            # 2. Get existing unit to check hash
            existing_unit = await self.storage.get_unit(unit.id)
            
            should_distill = force_distill
            if not existing_unit:
                should_distill = True
                logger.info("New unit discovered: %s", unit.name)
            elif existing_unit.code_hash != unit.code_hash:
                should_distill = True
                logger.info("Unit %s changed (hash mismatch)", unit.name)
            elif not existing_unit.summary:
                should_distill = True
                logger.info("Summary missing for %s", unit.name)
            
            if should_distill:
                async with self.semaphore:
                    logger.info("Distilling summary for %s...", unit.id)
                    try:
                        unit.summary = await self.intelligence.summarize(raw_code, unit.name)
                    except Exception as e:
                        logger.error("Failed to distill %s: %s", unit.name, e)
                        # Keep old summary if available, otherwise stay None
                        unit.summary = existing_unit.summary if existing_unit else None
            else:
                # Reuse existing summary if code hasn't changed
                unit.summary = existing_unit.summary if existing_unit else None
            
            # 3. Save to storage (includes embedding generation)
            await self.storage.upsert_unit(unit)

    async def search(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        """
        Performs semantic search across all indexed units.
        """
        return await self.storage.search_units(query, limit=limit)

    async def sync_project(self, paths: List[str], force_distill: bool = False):
        """
        Concurrent synchronization of multiple files.
        """
        tasks = [self.sync_file(p, force_distill=force_distill) for p in paths]
        await asyncio.gather(*tasks)
