import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from code_rag.core.manager import CodeRAGManager


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.upsert_unit = AsyncMock()
    storage.get_unit = AsyncMock(return_value=None)
    storage.search_units = AsyncMock(return_value=[])
    storage.delete_stale_units = AsyncMock()
    storage.set_dependency_path = AsyncMock()
    storage.close = AsyncMock()
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


@pytest.fixture
def manager(mock_storage, mock_parser, mock_intelligence):
    return CodeRAGManager(mock_storage, mock_parser, mock_intelligence)


class TestManagerExtra:
    """Extra tests to fill remaining coverage in manager.py."""

    @pytest.mark.asyncio
    async def test_sync_maven_failed_process(self, manager, tmp_path):
        """Test Maven sync handles non-zero return code."""
        (tmp_path / "pom.xml").write_text("<project/>")
        with patch("shutil.which", return_value="mvn"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"error logs")
            mock_exec.return_value = mock_proc

            await manager._sync_maven(tmp_path)
            # Should not crash, just log error

    @pytest.mark.asyncio
    async def test_sync_gradle_failed_process(self, manager, tmp_path):
        """Test Gradle sync handles non-zero return code."""
        (tmp_path / "build.gradle").touch()
        with patch("shutil.which", return_value="gradle"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"gradle error")
            mock_exec.return_value = mock_proc

            await manager._sync_gradle(tmp_path)

    @pytest.mark.asyncio
    async def test_sync_file_exception(self, manager, mock_parser):
        """Test sync_file handles parser exceptions."""
        mock_parser.distill_file.side_effect = Exception("Parser error")
        with pytest.raises(Exception):
            await manager.sync_file("broken.py")

    @pytest.mark.asyncio
    async def test_sync_project_worker_exception(self, manager, mock_parser):
        """Test worker handles individual file exceptions without crashing pool."""
        mock_parser.distill_file.side_effect = [Exception("F1 error"), []]
        await manager.sync_project(["f1.py", "f2.py"])
        # Should complete both
        assert mock_parser.distill_file.call_count == 2
