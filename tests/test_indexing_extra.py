import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from code_rag.services.dependencies import _sync_gradle, _sync_maven
from code_rag.services.indexing import sync_file, sync_project
from tests.embedder_stubs import StubEmbedder


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.upsert_unit = AsyncMock()
    storage.get_unit = AsyncMock(return_value=None)
    storage.search_units = AsyncMock(return_value=[])
    storage.delete_stale_units = AsyncMock()
    storage.set_dependency_path = AsyncMock()
    storage.close = AsyncMock()
    storage.has_embedding = AsyncMock(return_value=False)
    storage.list_units = AsyncMock(return_value=[])
    storage.mark_embedding_model_synced = AsyncMock()
    storage.embedding_model_dirty = False
    storage.embedder = StubEmbedder()
    return storage


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[])
    return parser


@pytest.fixture
def mock_intelligence():
    intel = MagicMock()
    intel.summarize = AsyncMock(return_value="summary")
    return intel


class TestIndexingExtra:
    """Extra tests for indexing and dependency helpers."""

    @pytest.mark.asyncio
    async def test_sync_maven_failed_process(self, mock_storage, tmp_path):
        """Test Maven sync handles non-zero return code."""
        (tmp_path / "pom.xml").write_text("<project/>")
        with patch("shutil.which", return_value="mvn"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"error logs")
            mock_exec.return_value = mock_proc

            await _sync_maven(mock_storage, tmp_path)

    @pytest.mark.asyncio
    async def test_sync_gradle_failed_process(self, mock_storage, tmp_path):
        """Test Gradle sync handles non-zero return code."""
        (tmp_path / "build.gradle").touch()
        with patch("shutil.which", return_value="gradle"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"gradle error")
            mock_exec.return_value = mock_proc

            await _sync_gradle(mock_storage, tmp_path)

    @pytest.mark.asyncio
    async def test_sync_file_exception(
        self, mock_storage, mock_parser, mock_intelligence
    ):
        """Test sync_file handles parser exceptions."""
        mock_parser.distill_file.side_effect = Exception("Parser error")
        with pytest.raises(Exception):
            await sync_file(mock_storage, mock_parser, mock_intelligence, "broken.py")

    @pytest.mark.asyncio
    async def test_sync_project_worker_exception(
        self, mock_storage, mock_parser, mock_intelligence
    ):
        """Test worker handles individual file exceptions without crashing pool."""

        async def distill(path):
            if path == "f1.py":
                raise Exception("F1 error")
            return []

        mock_parser.distill_file.side_effect = distill
        failures = await sync_project(
            mock_storage, mock_parser, mock_intelligence, ["f1.py", "f2.py"]
        )
        assert mock_parser.distill_file.call_count == 2
        assert failures == [("f1.py", "F1 error")]
