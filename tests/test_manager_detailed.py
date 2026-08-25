import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch

from code_rag.core.manager import CodeRAGManager
from code_rag.core.interfaces import IStorage, IParser, IIntelligence
from code_rag.core.models import KnowledgeUnit, UnitKind


@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=IStorage)
    storage.upsert_unit = AsyncMock()
    storage.get_unit = AsyncMock(return_value=None)
    storage.search_units = AsyncMock(return_value=[])
    storage.delete_stale_units = AsyncMock()
    storage.set_dependency_path = AsyncMock()
    storage.close = AsyncMock()
    return storage


@pytest.fixture
def mock_parser():
    parser = MagicMock(spec=IParser)
    parser.distill_file = AsyncMock(return_value=[])
    return parser


@pytest.fixture
def mock_intelligence():
    intel = MagicMock(spec=IIntelligence)
    intel.summarize = AsyncMock(return_value="distilled summary")
    return intel


@pytest.fixture
def manager(mock_storage, mock_parser, mock_intelligence):
    return CodeRAGManager(mock_storage, mock_parser, mock_intelligence)


@pytest.fixture
def manager_with_build_execution(mock_storage, mock_parser, mock_intelligence):
    return CodeRAGManager(
        mock_storage,
        mock_parser,
        mock_intelligence,
        allow_build_execution=True,
    )


class TestManagerDetailed:
    """Detailed unit tests for CodeRAGManager to increase coverage."""

    @pytest.mark.asyncio
    async def test_sync_dependencies_maven(self, manager_with_build_execution, tmp_path):
        """Test Maven dependency synchronization."""
        (tmp_path / "pom.xml").write_text("<project></project>")

        with patch("shutil.which", return_value="/usr/bin/mvn"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            # Mock classpath file creation
            cp_file = tmp_path / ".coderag_cp.txt"
            cp_file.write_text(f"lib1.jar{os.pathsep}lib2-1.0.jar")

            await manager_with_build_execution.sync_dependencies(str(tmp_path))

            # Verify storage calls
            assert (
                manager_with_build_execution.storage.set_dependency_path.call_count == 2
            )
            manager_with_build_execution.storage.set_dependency_path.assert_any_call(
                "lib1", "lib1.jar"
            )
            manager_with_build_execution.storage.set_dependency_path.assert_any_call(
                "lib2", "lib2-1.0.jar"
            )

    @pytest.mark.asyncio
    async def test_sync_dependencies_gradle(self, manager_with_build_execution, tmp_path):
        """Test Gradle dependency synchronization."""
        (tmp_path / "build.gradle").write_text("apply plugin: 'java'")

        with patch("shutil.which", return_value="/usr/bin/gradle"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_process = AsyncMock()
            # Gradle -q output with our marker
            mock_process.communicate.return_value = (b"CODERAG_CP:lib-gradle.jar", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await manager_with_build_execution.sync_dependencies(str(tmp_path))

            manager_with_build_execution.storage.set_dependency_path.assert_called_with(
                "lib-gradle", "lib-gradle.jar"
            )

    @pytest.mark.asyncio
    async def test_sync_dependencies_skips_without_build_files(
        self, manager, tmp_path, caplog
    ):
        """Non-Java projects should not warn when build execution is disabled."""
        with caplog.at_level("WARNING"):
            await manager.sync_dependencies(str(tmp_path))

        assert not caplog.records
        manager.storage.set_dependency_path.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_dependencies_blocked_without_opt_in(
        self, manager, tmp_path, caplog
    ):
        """Maven/Gradle sync requires explicit opt-in for trusted projects."""
        (tmp_path / "pom.xml").write_text("<project></project>")

        with caplog.at_level("WARNING"):
            await manager.sync_dependencies(str(tmp_path))

        assert any("Dependency sync is disabled by default" in r.message for r in caplog.records)
        manager.storage.set_dependency_path.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_file_delta_logic(self, manager, mock_storage, mock_parser):
        """Test sync_file with delta distillation logic."""
        unit = KnowledgeUnit(
            id="file.py:func",
            name="func",
            kind=UnitKind.FUNCTION,
            path="file.py",
            code_hash="new_hash",
            metadata={"raw_code": "def func(): pass"},
        )
        mock_parser.distill_file.return_value = [unit]

        # Scenario 1: Existing unit with same hash (Skip distillation)
        existing = KnowledgeUnit(
            id="file.py:func",
            name="func",
            kind=UnitKind.FUNCTION,
            path="file.py",
            code_hash="new_hash",
            summary="old summary",
        )
        mock_storage.get_unit.return_value = existing

        await manager.sync_file("file.py")
        assert unit.summary == "old summary"
        manager.intelligence.summarize.assert_not_called()

        # Scenario 2: Hash mismatch (Re-distill)
        unit.code_hash = "different_hash"
        await manager.sync_file("file.py")
        assert unit.summary == "distilled summary"
        manager.intelligence.summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_project_worker_pool(self, manager, mock_parser):
        """Test sync_project concurrent execution."""
        paths = ["f1.py", "f2.py", "f3.py"]
        mock_parser.distill_file.side_effect = [
            [
                KnowledgeUnit(
                    id="1",
                    name="n1",
                    kind=UnitKind.FUNCTION,
                    path="f1.py",
                    code_hash="h1",
                )
            ],
            [
                KnowledgeUnit(
                    id="2",
                    name="n2",
                    kind=UnitKind.FUNCTION,
                    path="f2.py",
                    code_hash="h2",
                )
            ],
            [
                KnowledgeUnit(
                    id="3",
                    name="n3",
                    kind=UnitKind.FUNCTION,
                    path="f3.py",
                    code_hash="h3",
                )
            ],
        ]

        await manager.sync_project(paths)

        assert mock_parser.distill_file.call_count == 3
        assert manager.storage.upsert_unit.call_count == 3

    @pytest.mark.asyncio
    async def test_manager_close(self, manager):
        """Test resource release on close."""
        # Add close method to intelligence if missing (for mock)
        manager.intelligence.close = MagicMock()

        await manager.close()

        manager.storage.close.assert_called_once()
        manager.intelligence.close.assert_called_once()
