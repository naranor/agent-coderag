"""
Custom exception hierarchy for CodeRAG.
"""

from typing import Optional

from .error_codes import ErrorCode


class CodeRAGError(Exception):
    """Base class for all CodeRAG exceptions."""

    def __init__(self, message: str = "", *, code: Optional[ErrorCode] = None):
        super().__init__(message)
        self.code = code


class StorageError(CodeRAGError):
    """Raised when a storage operation fails."""


class StorageBusyError(StorageError):
    """Raised when storage is locked or busy."""

    def __init__(self, message: str = "Storage is busy"):
        super().__init__(message, code=ErrorCode.STORAGE_BUSY)


class ParserError(CodeRAGError):
    """Raised when code parsing fails."""


class DiscoveryError(CodeRAGError):
    """Raised when dependency or API discovery fails."""


class IntelligenceError(CodeRAGError):
    """Raised when LLM or embedding operations fail."""


class SecurityError(CodeRAGError):
    """Raised when a security validation fails."""
