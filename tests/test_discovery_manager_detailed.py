import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from code_rag.discovery.manager import DiscoveryManager
from code_rag.core.exceptions import DiscoveryError


class TestDiscoveryManagerDetailed:
    """Detailed tests for DiscoveryManager."""

    @pytest.mark.asyncio
    async def test_extract_api_error_mapping(self):
        # Mock storage
        mock_storage = MagicMock()
        manager = DiscoveryManager(storage=mock_storage)

        # Scenario 1: Unknown language
        with pytest.raises(DiscoveryError) as exc:
            await manager.extract_api("lib", language="unsupported_lang")
        assert "No discovery provider registered" in str(exc.value)

        # Scenario 2: Provider raises generic exception, should be mapped to DiscoveryError
        # We patch the provider class inside the module to inject a failing mock
        with patch("code_rag.discovery.manager.PythonDiscoveryProvider") as mock_cls:
            mock_provider = mock_cls.return_value
            mock_provider.extract_api = AsyncMock(
                side_effect=RuntimeError("Hard crash")
            )

            # Re-init manager to pick up mocked provider
            manager = DiscoveryManager(storage=mock_storage)

            with pytest.raises(DiscoveryError) as exc:
                await manager.extract_api("lib", language="python")
            assert "API extraction failed" in str(exc.value)
            assert "Hard crash" in str(exc.value)
