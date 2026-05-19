import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from code_rag.discovery.providers.csharp import CSharpDiscoveryProvider
from code_rag.core.models import KnowledgeUnit, UnitKind


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[])
    return parser


class TestCSharpDiscoveryDetailed:
    """Detailed tests for CSharpDiscoveryProvider."""

    @pytest.mark.asyncio
    async def test_extract_api_csharp_basic(self, mock_parser, tmp_path):
        provider = CSharpDiscoveryProvider(parser=mock_parser)

        # Scenario: No .sln or .csproj found
        with patch(
            "code_rag.discovery.providers.csharp.find_directory_upwards",
            return_value=None,
        ):
            report = await provider.extract_api("testlib")
            assert "Could not find Solution" in report

    @pytest.mark.asyncio
    async def test_extract_api_csharp_with_csproj(self, mock_parser, tmp_path):
        provider = CSharpDiscoveryProvider(parser=mock_parser)

        csproj = tmp_path / "test.csproj"
        csproj.touch()
        (tmp_path / "Program.cs").touch()

        mock_parser.distill_file.return_value = [
            KnowledgeUnit(
                id="u1",
                name="ApiClass",
                kind=UnitKind.CLASS,
                path=str(tmp_path / "Program.cs"),
                code_hash="h",
                signature="public class ApiClass",
            )
        ]

        with patch(
            "code_rag.discovery.providers.csharp.find_directory_upwards",
            return_value=csproj,
        ):
            report = await provider.extract_api("testlib")
            assert "ApiClass" in report
            assert "CSharp Project" in report
