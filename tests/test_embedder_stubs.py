import pytest

from code_rag.core.exceptions import IntelligenceError
from tests.embedder_stubs import StubEmbedder, UnboundStubEmbedder


def test_stub_unbound_dimension_raises():
    stub = StubEmbedder(dim=None)  # type: ignore[arg-type]
    with pytest.raises(IntelligenceError, match="not bound"):
        _ = stub.dimension


def test_stub_bind_rejects_invalid_and_mismatch():
    stub = StubEmbedder(dim=8)
    with pytest.raises(IntelligenceError, match="positive int"):
        stub.bind_dimension(0)
    with pytest.raises(IntelligenceError, match="already bound"):
        stub.bind_dimension(4)
    stub.bind_dimension(8)


def test_unbound_stub_bind_mismatch():
    stub = UnboundStubEmbedder(probe_dim=8)
    stub.bind_dimension(8)
    with pytest.raises(IntelligenceError, match="already bound"):
        stub.bind_dimension(16)
