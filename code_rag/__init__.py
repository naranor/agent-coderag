"""Public package surface. Imports stay lazy so ``agent-coderag --help`` does not load DuckDB, ONNX, or LiteLLM."""

import importlib

# Names are assigned so static checkers accept ``__all__``, then removed so
# attribute access goes through ``__getattr__`` and stays lazy.
CodeRAG = None
SyncResult = None
ApiReport = None
SetupResult = None
KnowledgeUnit = None
UnitKind = None
DistillerConfig = None
CodeRAGError = None
ErrorCode = None
StorageBusyError = None
default_db_path = None

__all__ = [
    "CodeRAG",
    "SyncResult",
    "ApiReport",
    "SetupResult",
    "KnowledgeUnit",
    "UnitKind",
    "DistillerConfig",
    "CodeRAGError",
    "ErrorCode",
    "StorageBusyError",
    "default_db_path",
]

_EXPORTS = {
    "CodeRAG": ("code_rag.api.client", "CodeRAG"),
    "SyncResult": ("code_rag.api.models", "SyncResult"),
    "ApiReport": ("code_rag.api.models", "ApiReport"),
    "SetupResult": ("code_rag.api.models", "SetupResult"),
    "KnowledgeUnit": ("code_rag.core.models", "KnowledgeUnit"),
    "UnitKind": ("code_rag.core.models", "UnitKind"),
    "DistillerConfig": ("code_rag.intelligence.distiller", "DistillerConfig"),
    "CodeRAGError": ("code_rag.core.exceptions", "CodeRAGError"),
    "ErrorCode": ("code_rag.core.error_codes", "ErrorCode"),
    "StorageBusyError": ("code_rag.core.exceptions", "StorageBusyError"),
    "default_db_path": ("code_rag.paths", "default_db_path"),
}

del (
    CodeRAG,
    SyncResult,
    ApiReport,
    SetupResult,
    KnowledgeUnit,
    UnitKind,
    DistillerConfig,
    CodeRAGError,
    ErrorCode,
    StorageBusyError,
    default_db_path,
)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value
