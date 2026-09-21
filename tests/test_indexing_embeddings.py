from unittest.mock import AsyncMock, MagicMock

import pytest

from code_rag.core.constants import EMBEDDING_BATCH_SIZE
from code_rag.core.error_codes import ErrorCode
from code_rag.core.exceptions import IntelligenceError, StorageError
from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.services.indexing import (
    IndexStack,
    sync_file,
    sync_project,
    unit_embedding_text,
)
from tests.embedder_stubs import StubEmbedder


def _unit(**kwargs):
    data = dict(
        id="u1",
        name="n",
        kind=UnitKind.FUNCTION,
        path="f.py",
        code_hash="h1",
        summary="sum",
    )
    data.update(kwargs)
    return KnowledgeUnit(**data)


def _stack(storage, parser=None, intel=None):
    if parser is None:
        parser = MagicMock()
        parser.distill_file = AsyncMock(return_value=[])
    if intel is None:
        intel = MagicMock()
        intel.summarize = AsyncMock(return_value="sum")
    return IndexStack(storage, parser, intel)


def test_unit_embedding_text_fallback():
    u = _unit(summary=None, signature="sig()", docstring="d")
    assert unit_embedding_text(u) == "function n sig() d"


@pytest.mark.asyncio
async def test_skip_reembed_when_hash_summary_and_row_unchanged():
    stub = StubEmbedder(dim=4)
    storage = MagicMock()
    storage.embedder = stub
    storage.embedding_model_dirty = False
    existing = _unit()
    storage.get_unit = AsyncMock(return_value=existing)
    storage.has_embedding = AsyncMock(return_value=True)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()
    incoming = _unit(metadata={"raw_code": "def n(): pass"})
    parser.distill_file = AsyncMock(return_value=[incoming])
    stack = _stack(storage, parser=parser)
    await sync_file(stack, "f.py")
    storage.upsert_unit.assert_awaited()
    kwargs = storage.upsert_unit.await_args
    assert kwargs.args[0].id == "u1"
    vector = kwargs.kwargs.get(
        "vector", kwargs.args[1] if len(kwargs.args) > 1 else None
    )
    assert vector is None
    assert stub.aembed_calls == []


@pytest.mark.asyncio
async def test_reembed_on_hash_change_batches():
    stub = StubEmbedder(dim=4)
    storage = MagicMock()
    storage.embedder = stub
    storage.embedding_model_dirty = False
    storage.get_unit = AsyncMock(return_value=_unit(code_hash="old"))
    storage.has_embedding = AsyncMock(return_value=True)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()
    units = [
        _unit(id=f"u{i}", name=f"n{i}", code_hash="new", metadata={"raw_code": "x"})
        for i in range(EMBEDDING_BATCH_SIZE + 1)
    ]
    parser.distill_file = AsyncMock(return_value=units)
    intel = MagicMock()
    intel.summarize = AsyncMock(return_value="sum")
    stack = _stack(storage, parser=parser, intel=intel)
    await sync_file(stack, "f.py")
    assert len(stub.aembed_calls) == 2
    assert len(stub.aembed_calls[0]) == EMBEDDING_BATCH_SIZE
    assert len(stub.aembed_calls[1]) == 1
    assert storage.upsert_unit.await_count == EMBEDDING_BATCH_SIZE + 1
    assert (
        storage.upsert_unit.await_args.kwargs.get("vector") is not None
        or len(storage.upsert_unit.await_args.args) > 1
    )


@pytest.mark.asyncio
async def test_incremental_dirty_raises_storage_error():
    storage = MagicMock()
    storage.embedding_model_dirty = True
    storage.embedder = StubEmbedder()
    stack = _stack(storage)
    with pytest.raises(StorageError, match="sync --all"):
        await sync_file(stack, "f.py")


@pytest.mark.asyncio
async def test_sync_project_dirty_without_index_all_raises():
    storage = MagicMock()
    storage.embedding_model_dirty = True
    storage.embedder = StubEmbedder()
    storage.list_units = AsyncMock()
    stack = _stack(storage)
    with pytest.raises(StorageError, match="sync --all"):
        await sync_project(stack, ["src/a.py"])
    storage.list_units.assert_not_called()


@pytest.mark.asyncio
async def test_sync_project_empty_index_all_reembeds_dirty():
    stub = StubEmbedder(dim=4)
    storage = MagicMock()
    storage.embedder = stub
    storage.embedding_model_dirty = True
    storage.list_units = AsyncMock(return_value=[_unit(id="a", summary="sa")])
    storage.upsert_unit = AsyncMock()
    storage.mark_embedding_model_synced = AsyncMock()
    stack = _stack(storage)
    await sync_project(stack, [], index_all=True)
    storage.mark_embedding_model_synced.assert_awaited_once()
    assert stub.aembed_calls == [["sa"]]
    storage.upsert_unit.assert_awaited()


@pytest.mark.asyncio
async def test_sync_project_dirty_reembeds_all_then_file_sync_skips():
    stub = StubEmbedder(dim=4)
    storage = MagicMock()
    storage.embedder = stub
    storage.embedding_model_dirty = True
    u_a = _unit(id="a", summary="sa")
    u_b = _unit(id="b", summary="sb")
    storage.list_units = AsyncMock(return_value=[u_a, u_b])
    storage.upsert_unit = AsyncMock()
    storage.mark_embedding_model_synced = AsyncMock(
        side_effect=lambda: setattr(storage, "embedding_model_dirty", False)
    )
    storage.get_unit = AsyncMock(return_value=u_a)
    storage.has_embedding = AsyncMock(return_value=True)
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()
    parser.distill_file = AsyncMock(
        return_value=[_unit(id="a", summary="sa", metadata={"raw_code": "x"})]
    )
    stack = _stack(storage, parser=parser)
    await sync_project(stack, ["f.py"], index_all=True)
    storage.mark_embedding_model_synced.assert_awaited_once()
    assert stub.aembed_calls == [["sa", "sb"]]
    file_upserts = [
        c
        for c in storage.upsert_unit.await_args_list
        if c.args and getattr(c.args[0], "id", None) == "a"
    ]
    assert file_upserts
    last_a = file_upserts[-1]
    vector = last_a.kwargs.get("vector")
    assert vector is None


@pytest.mark.asyncio
async def test_concurrent_sync_file_does_not_overlap_aembed():
    import asyncio as aio
    from unittest.mock import patch

    from code_rag.intelligence.openai_embedder import OpenAICompatEmbedder

    storage = MagicMock()
    storage.embedding_model_dirty = False
    storage.get_unit = AsyncMock(return_value=None)
    storage.has_embedding = AsyncMock(return_value=False)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()

    real = OpenAICompatEmbedder(api_base="http://e", model="m")
    real.bind_dimension(2)
    in_flight = 0
    max_in_flight = 0

    async def fake(**kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await aio.sleep(0.05)
        in_flight -= 1
        item = MagicMock()
        item.embedding = [1.0, 0.0]
        resp = MagicMock()
        resp.data = [item]
        return resp

    storage.embedder = real
    intel = MagicMock()
    intel.summarize = AsyncMock(return_value="sum")
    parser = MagicMock()

    async def distill(path):
        return [_unit(id=path, metadata={"raw_code": "x"})]

    parser.distill_file = AsyncMock(side_effect=distill)
    with patch("litellm.aembedding", new=fake):
        await sync_project(IndexStack(storage, parser, intel), ["a.py", "b.py"])
    assert max_in_flight == 1


@pytest.mark.asyncio
async def test_sync_file_propagates_embedding_mismatch_code():
    storage = MagicMock()
    storage.embedding_model_dirty = False
    storage.embedder = StubEmbedder(dim=8)
    storage.get_unit = AsyncMock(return_value=None)
    storage.has_embedding = AsyncMock(return_value=False)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    storage.ensure_embeddings_bound = AsyncMock(
        side_effect=StorageError(
            "Embedding dimension mismatch (index=384, embedder=8). Run rebuild.",
            code=ErrorCode.EMBEDDING_MISMATCH,
        )
    )
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[_unit(metadata={"raw_code": "x"})])
    stack = _stack(storage, parser=parser)
    with pytest.raises(StorageError, match="Run rebuild") as ei:
        await sync_file(stack, "f.py")
    assert ei.value.code is ErrorCode.EMBEDDING_MISMATCH


@pytest.mark.asyncio
async def test_embed_and_upsert_count_mismatch():
    class ShortStub(StubEmbedder):
        async def aembed(self, texts):
            return []

    storage = MagicMock()
    storage.embedding_model_dirty = False
    storage.embedder = ShortStub(dim=4)
    storage.get_unit = AsyncMock(return_value=None)
    storage.has_embedding = AsyncMock(return_value=False)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[_unit(metadata={"raw_code": "x"})])
    stack = _stack(storage, parser=parser)
    with pytest.raises(IntelligenceError, match="count mismatch"):
        await sync_file(stack, "f.py")


@pytest.mark.asyncio
async def test_sync_file_distills_when_summary_missing():
    storage = MagicMock()
    storage.embedding_model_dirty = False
    storage.embedder = StubEmbedder(dim=4)
    existing = _unit(summary=None)
    storage.get_unit = AsyncMock(return_value=existing)
    storage.has_embedding = AsyncMock(return_value=True)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()
    parser.distill_file = AsyncMock(
        return_value=[_unit(summary=None, metadata={"raw_code": "x"})]
    )
    intel = MagicMock()
    intel.summarize = AsyncMock(return_value="filled")
    stack = _stack(storage, parser=parser, intel=intel)
    await sync_file(stack, "f.py")
    intel.summarize.assert_awaited_once()
    stored = storage.upsert_unit.await_args.args[0]
    assert stored.summary == "filled"


@pytest.mark.asyncio
async def test_sync_file_distill_failure_keeps_existing_summary():
    storage = MagicMock()
    storage.embedding_model_dirty = False
    storage.embedder = StubEmbedder(dim=4)
    existing = _unit(summary="old", code_hash="old")
    storage.get_unit = AsyncMock(return_value=existing)
    storage.has_embedding = AsyncMock(return_value=True)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()
    parser = MagicMock()
    parser.distill_file = AsyncMock(
        return_value=[_unit(summary=None, code_hash="new", metadata={"raw_code": "x"})]
    )
    intel = MagicMock()
    intel.summarize = AsyncMock(side_effect=RuntimeError("llm down"))
    stack = _stack(storage, parser=parser, intel=intel)
    await sync_file(stack, "f.py")
    stored = storage.upsert_unit.await_args.args[0]
    assert stored.summary == "old"
