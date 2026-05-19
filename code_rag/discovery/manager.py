import logging
from typing import Dict, Optional

from .providers.base import IDiscoveryProvider
from .providers.python import PythonDiscoveryProvider
from .providers.java import JavaDiscoveryProvider
from .providers.go import GoDiscoveryProvider
from .providers.javascript import JavaScriptDiscoveryProvider
from .providers.rust import RustDiscoveryProvider
from .providers.csharp import CSharpDiscoveryProvider

from ..core.interfaces import IStorage
from ..core.exceptions import DiscoveryError

logger = logging.getLogger(__name__)


class DiscoveryManager:
    """Dispatches discovery requests to language-specific providers."""

    def __init__(self, storage: Optional[IStorage] = None) -> None:
        self._providers: Dict[str, IDiscoveryProvider] = {}
        self.storage = storage
        # Auto-register default providers
        self.register_provider("python", PythonDiscoveryProvider())
        self.register_provider("java", JavaDiscoveryProvider(storage=storage))
        self.register_provider("go", GoDiscoveryProvider())
        self.register_provider("javascript", JavaScriptDiscoveryProvider())
        self.register_provider("typescript", JavaScriptDiscoveryProvider())
        self.register_provider("rust", RustDiscoveryProvider())
        self.register_provider("csharp", CSharpDiscoveryProvider())
        self.register_provider("c_sharp", CSharpDiscoveryProvider())

    def register_provider(self, language: str, provider: IDiscoveryProvider):
        """Registers a new discovery provider."""
        self._providers[language.lower()] = provider
        logger.debug("Registered discovery provider for %s", language)

    async def extract_api(self, library_name: str, language: str) -> str:
        """
        Extracts API for a library using the specified language provider.
        """
        provider = self._providers.get(language.lower())
        if not provider:
            supported = ", ".join(self._providers.keys())
            raise DiscoveryError(
                f"No discovery provider registered for language '{language}'. "
                f"Supported: {supported}"
            )

        logger.info("Extracting API for '%s' using %s provider", library_name, language)
        try:
            return await provider.extract_api(library_name)
        except Exception as e:
            if isinstance(e, DiscoveryError):
                raise
            raise DiscoveryError(
                f"API extraction failed for '{library_name}' ({language}): {e}"
            ) from e
