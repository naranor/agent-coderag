import logging
from .manager import DiscoveryManager

logger = logging.getLogger(__name__)

# Global manager instance
_manager = DiscoveryManager()


async def extract_library_api(library_name: str, language: str) -> str:
    """
    Extracts the public API (classes, methods) of an installed library.
    Delegates to DiscoveryManager which uses language-specific providers.
    """
    return await _manager.extract_api(library_name, language=language)
