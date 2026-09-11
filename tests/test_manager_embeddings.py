from unittest.mock import AsyncMock, MagicMock

import pytest

from code_rag.core.constants import EMBEDDING_BATCH_SIZE
from code_rag.core.exceptions import StorageError
from code_rag.core.manager import CodeRAGManager, unit_embedding_text
from code_rag.core.models import KnowledgeUnit, UnitKind
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


def _manager(storage, parser=None, intel=None):
    if parser is None:
        parser = MagicMock()
        parser.distill_file = AsyncMock(return_value=[])
    if intel is None:
        intel = MagicMock()
        intel.summarize = AsyncMock(return_value="sum")
    return CodeRAGManager(storage, parser, intel)


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
    manager = _manager(storage, parser=parser)
    await manager.sync_file("f.py")
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
    manager = _manager(storage, parser=parser, intel=intel)
    await manager.sync_file("f.py")
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
    manager = _manager(storage)
    with pytest.raises(StorageError, match="sync --all"):
        await manager.sync_file("f.py")


@pytest.mark.asyncio
async def test_sync_project_dirty_without_index_all_raises():
    storage = MagicMock()
    storage.embedding_model_dirty = True
    storage.embedder = StubEmbedder()
    storage.list_units = AsyncMock()
    manager = _manager(storage)
    with pytest.raises(StorageError, match="sync --all"):
        await manager.sync_project(["src/a.py"])
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
    manager = _manager(storage)
    await manager.sync_project([], index_all=True)
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
    manager = _manager(storage, parser=parser)
    await manager.sync_project(["f.py"], index_all=True)
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
    if vector is None and len(last_a.args) > 1:
        vector = last_a.args[1]
    assert vector is None


@pytest.mark.asyncio
async def test_concurrent_sync_file_does_not_overlap_aembed():
    import asyncio as aio

    class SlowStub(StubEmbedder):
        def __init__(self):
            super().__init__(dim=2)
            self.max_in_flight = 0
            self._in = 0

        async def aembed(self, texts):
            self._in += 1
            self.max_in_flight = max(self.max_in_flight, self._in)
            await aio.sleep(0.05)
            self._in -= 1
            return await super().aembed(texts)

    stub = SlowStub()
    storage = MagicMock()
    storage.embedder = stub
    storage.embedding_model_dirty = False
    storage.get_unit = AsyncMock(return_value=None)
    storage.has_embedding = AsyncMock(return_value=False)
    storage.upsert_unit = AsyncMock()
    storage.delete_stale_units = AsyncMock()

    def make_parser(uid):
        p = MagicMock()
        p.distill_file = AsyncMock(
            return_value=[_unit(id=uid, metadata={"raw_code": "x"})]
        )
        return p

    from code_rag.intelligence.openai_embedder import OpenAICompatEmbedder
    from unittest.mock import patch

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
    manager = CodeRAGManager(storage, parser, intel)
    with patch("code_rag.intelligence.openai_embedder.litellm.aembedding", new=fake):
        await manager.sync_project(["a.py", "b.py"])
    assert max_in_flight == 1
