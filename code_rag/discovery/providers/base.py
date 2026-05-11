from abc import ABC, abstractmethod


class IDiscoveryProvider(ABC):
    """Base interface for API discovery providers."""

    @abstractmethod
    async def extract_api(self, library_name: str) -> str:
        """Extracts the public API (classes, methods) for the given library."""
