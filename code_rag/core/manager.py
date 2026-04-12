import logging
import asyncio
from typing import List, Optional, Dict, Any
from .interfaces import IStorage, IParser, IIntelligence
from .models import KnowledgeUnit, UnitKind

logger = logging.getLogger(__name__)

class CodeRAGManager:
    """
    Main orchestrator for the CodeRAG system.
    Coordinates parsing, semantic distillation, and storage.
    """
    
    def __init__(
        self, 
        storage: IStorage, 
        parser: IParser, 
        intelligence: IIntelligence
    ):
        self.storage = storage
        self.parser = parser
        self.intelligence = intelligence

    async def sync_file(self, path: str, force_distill: bool = False):
        """
        Synchronizes a file with the knowledge base.
        Checks hashes and triggers distillation for new/changed units.
        """
        logger.info(f"Syncing file: {path}")
        
        # 1. Parse current structure
        current_units = await self.parser.parse_file(path)
        if not current_units:
            return

        # 2. Get existing units from storage to compare hashes
        # We need a way to get units by path from storage
        # For simplicity, we'll assume search_units can filter or we'll add a method
        # In this implementation, we'll just upsert and let the logic decide
        
        for unit in current_units:
            # v5.40: Delta-distillation logic
            # Extract code from metadata (temporary)
            raw_code = unit.metadata.pop("raw_code", "")
            
            should_distill = force_distill
            
            # TODO: Check existing hash in DB before distilling
            # For now, we'll distill if force is True or summary is missing
            
            if should_distill:
                logger.info(f"Distilling summary for {unit.id}")
                unit.summary = await self.intelligence.summarize(raw_code, unit.name)
            
            # 3. Save to storage (includes embedding generation)
            await self.storage.upsert_unit(unit)

    async def search(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        """
        Performs semantic search across the knowledge base.
        """
        return await self.storage.search_units(query, limit=limit)

    async def sync_project(self, paths: List[str], force_distill: bool = False):
        """
        Concurrent synchronization of multiple files.
        """
        tasks = [self.sync_file(p, force_distill=force_distill) for p in paths]
        await asyncio.gather(*tasks)
