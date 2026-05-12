from abc import ABC, abstractmethod
from typing import List, Optional
from .models import KnowledgeUnit, Relation


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

    @abstractmethod
    async def upsert_relation(self, relation: Relation):
        pass

    @abstractmethod
    async def get_relations(
        self, unit_id: str, direction: str = "out"
    ) -> List[Relation]:
        pass

    @abstractmethod
    async def set_dependency_path(self, lib_name: str, path: str) -> None:
        """Caches the absolute path to a library's JAR/binary."""

    @abstractmethod
    async def get_dependency_path(self, lib_name: str) -> Optional[str]:
        """Retrieves the cached path for a library."""


class IIntelligence(ABC):
    """Interface for LLM-based analysis (distillation, embeddings)."""

    @abstractmethod
    async def summarize(self, code: str, unit_name: str) -> str:
        pass
