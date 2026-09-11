from abc import ABC, abstractmethod
from typing import List, Optional
from .models import KnowledgeUnit, Relation


class IParser(ABC):
    """Interface for extracting structure from code."""

    @abstractmethod
    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        pass


class IEmbedder(ABC):
    """Port for local ONNX and remote OpenAI-compatible embedders."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Bound embedding width. No IO. Raise if unbound."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable id written to index_meta (local:mini-lm or remote model)."""

    @abstractmethod
    def bind_dimension(self, dim: int) -> None:
        """Record the index dimension before search/upsert."""

    @abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Return len(texts) L2-normalized rows, 1:1 with input order."""

    @abstractmethod
    async def close(self) -> None:
        """Release ONNX session and/or HTTP resources."""


class IStorage(ABC):
    """Interface for storing the index."""

    @abstractmethod
    async def upsert_unit(
        self, unit: KnowledgeUnit, vector: Optional[List[float]] = None
    ):
        pass

    @abstractmethod
    async def has_embedding(self, unit_id: str) -> bool:
        pass

    @abstractmethod
    async def list_units(self) -> List[KnowledgeUnit]:
        pass

    @abstractmethod
    async def mark_embedding_model_synced(self) -> None:
        pass

    @property
    @abstractmethod
    def embedding_model_dirty(self) -> bool:
        pass

    @property
    @abstractmethod
    def embedder(self) -> IEmbedder:
        """Bound embedder used for vector search and upsert."""

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

    @abstractmethod
    async def delete_stale_units(
        self, file_path: str, current_unit_ids: List[str]
    ) -> None:
        """Removes units that are no longer present in the given file."""

    @abstractmethod
    async def close(self) -> None:
        """Closes the storage connection and releases resources."""


class IIntelligence(ABC):
    """Interface for LLM-based analysis (distillation, embeddings)."""

    @abstractmethod
    async def summarize(self, code: str, unit_name: str) -> str:
        pass
