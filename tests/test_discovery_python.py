import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from code_rag.discovery.providers.python import PythonDiscoveryProvider
from code_rag.core.models import KnowledgeUnit, UnitKind


@pytest.mark.asyncio
async def test_python_provider_prefers_stubs():
    """Test that provider prioritizes .pyi files."""
    mock_parser = MagicMock()
    mock_parser.distill_file = AsyncMock(
        return_value=[
            KnowledgeUnit(
                id="test:stub",
                name="stub_fn",
                kind=UnitKind.FUNCTION,
                signature="def stub_fn(): ...",
                path="test.pyi",
                code_hash="h1",
            )
        ]
    )

    provider = PythonDiscoveryProvider(parser=mock_parser)

    with patch.object(provider, "_find_library_root", return_value=Path("/fake/lib")):
        with patch("pathlib.Path.rglob", return_value=[Path("/fake/lib/api.pyi")]):
            result = await provider.extract_api("fake_lib")

            assert "Static Analysis" in result
            assert "stub_fn" in result
            mock_parser.distill_file.assert_called_once()


@pytest.mark.asyncio
async def test_python_provider_runtime_fallback():
    """Test that provider falls back to runtime when no files found."""
    mock_parser = MagicMock()
    provider = PythonDiscoveryProvider(parser=mock_parser)

    with patch.object(provider, "_find_library_root", return_value=None):
        with patch.object(
            provider, "_extract_runtime", return_value="Runtime API"
        ) as mock_runtime:
            result = await provider.extract_api("unknown_lib")
            assert result == "Runtime API"
            mock_runtime.assert_called_once_with("unknown_lib")


@pytest.mark.asyncio
async def test_python_provider_static_source_fallback():
    """Test that provider tries .py files if no .pyi found."""
    mock_parser = MagicMock()
    mock_parser.distill_file = AsyncMock(
        return_value=[
            KnowledgeUnit(
                id="test:src",
                name="src_fn",
                kind=UnitKind.FUNCTION,
                signature="def src_fn(): ...",
                path="init.py",
                code_hash="h2",
            )
        ]
    )

    provider = PythonDiscoveryProvider(parser=mock_parser)

    with patch.object(provider, "_find_library_root", return_value=Path("/fake/lib")):
        # Mock rglob for .pyi to return empty, and glob for .py to return files
        with patch("pathlib.Path.rglob", return_value=[]):
            with patch(
                "pathlib.Path.glob", return_value=[Path("/fake/lib/__init__.py")]
            ):
                result = await provider.extract_api("fake_lib")

                assert "Static Analysis" in result
                assert "src_fn" in result
                mock_parser.distill_file.assert_called_once()
