import pytest
from unittest.mock import patch
from code_rag.intelligence.distiller import Distiller, DistillerConfig


class TestDistillerExtra:
    """Extra tests for distiller coverage."""

    @pytest.mark.asyncio
    async def test_summarize_error_handling(self):
        config = DistillerConfig(api_key="test", model="test")
        distiller = Distiller(config)

        # Match litellm error behavior
        with patch("litellm.completion", side_effect=Exception("Connection error")):
            with pytest.raises(Exception) as exc:
                await distiller.summarize("code", "unit")
            assert "Connection error" in str(exc.value)

    def test_config_load_file_not_found(self, tmp_path):
        # Patch the global os module instead of trying to find it in distiller
        with patch(
            "os.path.exists",
            side_effect=lambda p: False if "distiller_config.json" in str(p) else True,
        ):
            config = DistillerConfig.load()
            assert config.provider == "openai"  # default
