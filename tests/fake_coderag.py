from unittest.mock import AsyncMock

from code_rag.api.models import ApiReport, SyncResult


def fake_coderag_class(**method_overrides):
    """Test double for CLI CodeRAG construction and method calls.

    Returns ``(FakeRAG, instances)`` so tests can inspect constructed objects.
    """
    instances: list = []
    success = SyncResult(status="success", indexed_files=1)

    class FakeRAG:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            instances.append(self)
            self.sync = AsyncMock(return_value=success)
            self.search = AsyncMock(return_value=[])
            self.api = AsyncMock(
                return_value=ApiReport(
                    library="lib", language="python", report="Public API..."
                )
            )
            self.rebuild = AsyncMock(return_value=success)
            self.close = AsyncMock()
            for name, value in method_overrides.items():
                setattr(self, name, value)

    return FakeRAG, instances
