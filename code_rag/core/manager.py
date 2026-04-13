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
            raw_code = unit.metadata.pop("raw_code", "")
            
            # 2. Get existing unit to check hash
            existing_unit = await self.storage.get_unit(unit.id)
            
            should_distill = force_distill
            if not existing_unit:
                should_distill = True
                logger.info(f"New unit discovered: {unit.name}")
            elif existing_unit.code_hash != unit.code_hash:
                should_distill = True
                logger.info(f"Unit {unit.name} changed (hash mismatch)")
            elif not existing_unit.summary:
                should_distill = True
                logger.info(f"Summary missing for {unit.name}")
            
            if should_distill:
                logger.info(f"Distilling summary for {unit.id}...")
                try:
                    unit.summary = await self.intelligence.summarize(raw_code, unit.name)
                except Exception as e:
                    logger.error(f"Failed to distill {unit.name}: {e}")
                    # Keep old summary if available, otherwise stay None
                    unit.summary = existing_unit.summary if existing_unit else None
            else:
                # Reuse existing summary if code hasn't changed
                unit.summary = existing_unit.summary if existing_unit else None
            
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
