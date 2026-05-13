import logging
from typing import Dict, Optional

from .providers.base import IDiscoveryProvider
from .providers.python import PythonDiscoveryProvider
from .providers.java import JavaDiscoveryProvider
from .providers.go import GoDiscoveryProvider
from .providers.javascript import JavaScriptDiscoveryProvider
from .providers.rust import RustDiscoveryProvider

from ..core.interfaces import IStorage

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

    def register_provider(self, language: str, provider: IDiscoveryProvider):
        """Registers a new discovery provider."""
        self._providers[language.lower()] = provider
        logger.debug("Registered discovery provider for %s", language)

    async def extract_api(
        self, library_name: str, language: Optional[str] = None
    ) -> str:
        """
        Extracts API for a library.
        If language is not provided, it tries all registered providers.
        """
        if language:
            provider = self._providers.get(language.lower())
            if provider:
                return await provider.extract_api(library_name)
            return f"Error: No discovery provider registered for language '{language}'"

        # Try all providers sequentially if no language specified
        # This maintains backward compatibility with the old dependency.py behavior
        results = []
        for lang, provider in self._providers.items():
            logger.info(
                "Attempting discovery for '%s' using %s provider", library_name, lang
            )
            result = await provider.extract_api(library_name)
            # Basic heuristic: if it doesn't start with 'Failed' or 'Error', it probably worked
            if not result.startswith("Failed") and not result.startswith("Error"):
                return result
            results.append(f"[{lang}] {result}")

        return "\n".join(results) or f"Error: Could not find API for '{library_name}'"
