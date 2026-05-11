import logging
from .base import IDiscoveryProvider
from ..java_discovery import extract_java_api

logger = logging.getLogger(__name__)


class JavaDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Java libraries using javap."""

    async def extract_api(self, library_name: str) -> str:
        """Extracts Java API using javap on found JAR files."""
        return await extract_java_api(library_name)
