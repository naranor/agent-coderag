from abc import ABC
from pathlib import Path

import pytest

from code_rag.core.constants import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_PROBE_TEXT,
    EMBEDDING_REQUEST_TIMEOUT,
    LOCAL_EMBEDDING_MODEL_ID,
)
from code_rag.core.interfaces import IEmbedder


def test_embedding_constants():
    assert EMBEDDING_DIM == 384
    assert EMBEDDING_BATCH_SIZE == 32
    assert EMBEDDING_REQUEST_TIMEOUT == 30
    assert EMBEDDING_PROBE_TEXT == "probe"
    assert LOCAL_EMBEDDING_MODEL_ID == "local:mini-lm"


def test_iembedder_is_abc():
    assert issubclass(IEmbedder, ABC)


def test_iembedder_incomplete_cannot_instantiate():
    class Incomplete(IEmbedder):
        pass

    with pytest.raises(TypeError):
        Incomplete()


@pytest.mark.asyncio
async def test_iembedder_subclass_roundtrip():
    class Dummy(IEmbedder):
        def __init__(self):
            self._dim = 8

        @property
        def dimension(self) -> int:
            return self._dim

        @property
        def model_id(self) -> str:
            return "stub"

        def bind_dimension(self, dim: int) -> None:
            self._dim = dim

        async def aembed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self._dim for _ in texts]

        async def close(self) -> None:
            return None

    dummy = Dummy()
    dummy.bind_dimension(4)
    assert dummy.dimension == 4
    assert dummy.model_id == "stub"
    rows = await dummy.aembed(["a", "b"])
    assert len(rows) == 2
    assert rows[0] == [0.0, 0.0, 0.0, 0.0]
    await dummy.close()


def test_core_modules_do_not_import_numpy():
    root = Path("code_rag/core")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import numpy" not in text
        assert "from numpy" not in text
