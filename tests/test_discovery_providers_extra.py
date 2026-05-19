import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from code_rag.discovery.providers.go import GoDiscoveryProvider
from code_rag.discovery.providers.java import JavaDiscoveryProvider
from code_rag.core.models import KnowledgeUnit, UnitKind


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[])
    return parser


class TestGoDiscoveryDetailed:
    """Detailed tests for GoDiscoveryProvider."""

    @pytest.mark.asyncio
    async def test_extract_api_go(self, mock_parser, tmp_path):
        provider = GoDiscoveryProvider(parser=mock_parser)
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module test")
        main_go = tmp_path / "main.go"
        main_go.touch()

        mock_parser.distill_file.return_value = [
            KnowledgeUnit(
                id="u1",
                name="ApiFunc",
                kind=UnitKind.FUNCTION,
                path=str(main_go),
                code_hash="h",
                signature="func ApiFunc()",
            )
        ]

        with patch(
            "code_rag.discovery.providers.go.find_directory_upwards",
            return_value=go_mod,
        ):
            report = await provider.extract_api("test")
            assert "ApiFunc" in report


class TestJavaDiscoveryDetailed:
    """Detailed tests for JavaDiscoveryProvider."""

    @pytest.mark.asyncio
    async def test_extract_api_java(self, mock_parser, tmp_path):
        # Mock storage for dependency path
        mock_storage = MagicMock()
        mock_storage.get_dependency_path = AsyncMock(
            return_value=str(tmp_path / "lib.jar")
        )

        provider = JavaDiscoveryProvider(storage=mock_storage, parser=mock_parser)

        # Test error path first
        mock_storage.get_dependency_path.return_value = None
        report = await provider.extract_api("missing")
        assert "Error" in report
