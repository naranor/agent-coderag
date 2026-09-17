import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.core.interfaces import IIntelligence, IParser, IStorage
from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.services.dependencies import sync_dependencies
from code_rag.services.indexing import sync_file, sync_project
from tests.embedder_stubs import StubEmbedder


@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=IStorage)
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
    parser = MagicMock(spec=IParser)
    parser.distill_file = AsyncMock(return_value=[])
    return parser


@pytest.fixture
def mock_intelligence():
    intel = MagicMock(spec=IIntelligence)
    intel.summarize = AsyncMock(return_value="distilled summary")
    return intel


class TestIndexingDetailed:
    """Detailed unit tests for indexing and dependency sync."""

    @pytest.mark.asyncio
    async def test_sync_dependencies_maven(self, mock_storage, tmp_path):
        """Test Maven dependency synchronization."""
        (tmp_path / "pom.xml").write_text("<project></project>")

        with patch("shutil.which", return_value="/usr/bin/mvn"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            cp_file = tmp_path / ".coderag_cp.txt"
            cp_file.write_text(f"lib1.jar{os.pathsep}lib2-1.0.jar")

            await sync_dependencies(
                mock_storage, str(tmp_path), allow_build_execution=True
            )

            assert mock_storage.set_dependency_path.call_count == 2
            mock_storage.set_dependency_path.assert_any_call("lib1", "lib1.jar")
            mock_storage.set_dependency_path.assert_any_call("lib2", "lib2-1.0.jar")

    @pytest.mark.asyncio
    async def test_sync_dependencies_gradle(self, mock_storage, tmp_path):
        """Test Gradle dependency synchronization."""
        (tmp_path / "build.gradle").write_text("apply plugin: 'java'")

        with patch("shutil.which", return_value="/usr/bin/gradle"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"CODERAG_CP:lib-gradle.jar", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await sync_dependencies(
                mock_storage, str(tmp_path), allow_build_execution=True
            )

            mock_storage.set_dependency_path.assert_called_with(
                "lib-gradle", "lib-gradle.jar"
            )

    @pytest.mark.asyncio
    async def test_sync_dependencies_non_java_does_not_warn(
        self, mock_storage, tmp_path, caplog
    ):
        """Test dependency synchronization stays quiet without build files."""
        with caplog.at_level(logging.WARNING):
            await sync_dependencies(mock_storage, str(tmp_path))

        assert not caplog.records

    @pytest.mark.asyncio
    async def test_sync_dependencies_requires_build_execution_opt_in(
        self, mock_storage, tmp_path, caplog
    ):
        """Test repository build files are not executed without explicit opt-in."""
        (tmp_path / "pom.xml").write_text("<project></project>")

        with caplog.at_level(logging.WARNING), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            await sync_dependencies(mock_storage, str(tmp_path))

        mock_exec.assert_not_called()
        assert "Dependency sync is disabled by default" in caplog.text

    @pytest.mark.asyncio
    async def test_sync_file_delta_logic(
        self, mock_storage, mock_parser, mock_intelligence
    ):
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

        existing = KnowledgeUnit(
            id="file.py:func",
            name="func",
            kind=UnitKind.FUNCTION,
            path="file.py",
            code_hash="new_hash",
            summary="old summary",
        )
        mock_storage.get_unit.return_value = existing

        await sync_file(mock_storage, mock_parser, mock_intelligence, "file.py")
        assert unit.summary == "old summary"
        mock_intelligence.summarize.assert_not_called()

        unit.code_hash = "different_hash"
        await sync_file(mock_storage, mock_parser, mock_intelligence, "file.py")
        assert unit.summary == "distilled summary"
        mock_intelligence.summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_project_worker_pool(
        self, mock_storage, mock_parser, mock_intelligence
    ):
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

        await sync_project(mock_storage, mock_parser, mock_intelligence, paths)

        assert mock_parser.distill_file.call_count == 3
        assert mock_storage.upsert_unit.call_count == 3

    @pytest.mark.asyncio
    async def test_storage_close_after_sync(
        self, mock_storage, mock_parser, mock_intelligence
    ):
        """Test storage can be closed after indexing (embedder owned by facade)."""
        await sync_file(mock_storage, mock_parser, mock_intelligence, "file.py")
        await mock_storage.close()
        mock_storage.close.assert_called_once()
