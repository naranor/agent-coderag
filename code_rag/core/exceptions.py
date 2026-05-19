"""
Custom exception hierarchy for CodeRAG.
"""


class CodeRAGError(Exception):
    """Base class for all CodeRAG exceptions."""


class StorageError(CodeRAGError):
    """Raised when a storage operation fails."""


class ParserError(CodeRAGError):
    """Raised when code parsing fails."""


class DiscoveryError(CodeRAGError):
    """Raised when dependency or API discovery fails."""


class IntelligenceError(CodeRAGError):
    """Raised when LLM or embedding operations fail."""


class SecurityError(CodeRAGError):
    """Raised when a security validation fails."""
