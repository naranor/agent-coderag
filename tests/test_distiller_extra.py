import pytest
from unittest.mock import patch
from code_rag.intelligence.distiller import Distiller, DistillerConfig


class TestDistillerExtra:
    """Extra tests for distiller coverage."""

    @pytest.mark.asyncio
    async def test_summarize_error_handling(self):
        config = DistillerConfig(
            api_key="test",
            model="test",
            provider="openai",
            api_base="http://localhost:8081/v1",
        )
        distiller = Distiller(config)

        # Match litellm error behavior
        with patch("litellm.acompletion", side_effect=Exception("Connection error")):
            with pytest.raises(Exception) as exc:
                await distiller.summarize("code", "unit")
            assert "Connection error" in str(exc.value)

    def test_config_load_file_not_found(self, tmp_path):
        with patch(
            "code_rag.intelligence.distiller.get_global_dir", return_value=tmp_path
        ):
            config = DistillerConfig.load()
            assert config.provider is None
            assert not config.is_llm_configured()
