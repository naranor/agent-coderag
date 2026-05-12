import logging
from typing import Optional
from .base import IDiscoveryProvider
from ..java_discovery import extract_java_api
from ...core.interfaces import IStorage

logger = logging.getLogger(__name__)


class JavaDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Java libraries using javap and build-system cache."""

    def __init__(self, storage: Optional[IStorage] = None):
        self.storage = storage

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts Java API using:
        1. Cached path from Maven/Gradle resolution (highest precision)
        2. Legacy heuristic search in .m2/.gradle (fallback)
        """
        if self.storage:
            cached_path = await self.storage.get_dependency_path(library_name)
            if cached_path:
                logger.info(
                    "Using cached JAR path for '%s': %s", library_name, cached_path
                )
                return await extract_java_api(library_name, jar_path=cached_path)

        logger.info(
            "No cached path for '%s', falling back to heuristic search", library_name
        )
        return await extract_java_api(library_name)
