from code_rag.core.exceptions import IntelligenceError
from code_rag.core.interfaces import IEmbedder


class StubEmbedder(IEmbedder):
    def __init__(self, dim: int = 384, model_id: str = "stub"):
        self._dim = dim
        self._model_id = model_id
        self.aembed_calls: list[list[str]] = []
        self.closed = False

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise IntelligenceError("Embedding dimension is not bound")
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def bind_dimension(self, dim: int) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise IntelligenceError("embedding dimension must be a positive int")
        if self._dim is not None and dim != self._dim:
            raise IntelligenceError(
                f"Cannot bind dimension {dim}; embedder already bound to {self._dim}"
            )
        self._dim = dim

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        self.aembed_calls.append(list(texts))
        return [[0.0] * self._dim for _ in texts]

    async def close(self) -> None:
        self.closed = True


class UnboundStubEmbedder(StubEmbedder):
    def __init__(self, probe_dim: int, model_id: str = "remote-model"):
        super().__init__(dim=probe_dim, model_id=model_id)
        self._bound = False
        self._probe_dim = probe_dim

    @property
    def dimension(self) -> int:
        if not self._bound:
            raise IntelligenceError("Embedding dimension is not bound")
        return self._dim

    def bind_dimension(self, dim: int) -> None:
        if self._bound and dim != self._dim:
            raise IntelligenceError(
                f"Cannot bind dimension {dim}; embedder already bound to {self._dim}"
            )
        self._dim = dim
        self._bound = True

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        self.aembed_calls.append(list(texts))
        width = self._dim if self._bound else self._probe_dim
        return [[0.0] * width for _ in texts]
