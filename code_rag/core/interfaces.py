from abc import ABC, abstractmethod
from typing import List, Optional
from .models import KnowledgeUnit

class IParser(ABC):
    """Interface for extracting structure from code."""
    @abstractmethod
    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        pass

class IStorage(ABC):
    """Interface for storing the index."""
    @abstractmethod
    async def upsert_unit(self, unit: KnowledgeUnit):
        pass
    
    @abstractmethod
    async def get_unit(self, unit_id: str) -> Optional[KnowledgeUnit]:
        pass
    
    @abstractmethod
    async def search_units(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        pass

class IIntelligence(ABC):
    """Interface for LLM-based analysis (distillation, embeddings)."""
    @abstractmethod
    async def summarize(self, code: str, unit_name: str) -> str:
        pass
