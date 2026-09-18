import pytest
from unittest.mock import MagicMock, patch

from code_rag.intelligence.distiller import Distiller, DistillerConfig


class TestDistillerConfig:
    """Tests for DistillerConfig."""

    def test_load_default_config(self, tmp_path):
        """Test loading default config from an isolated global dir."""
        with patch(
            "code_rag.intelligence.distiller.get_global_dir", return_value=tmp_path
        ):
            config = DistillerConfig.load()
        assert config.model is None
        assert config.provider is None
        assert config.api_base is None
        assert not config.is_llm_configured()

    def test_config_defaults(self):
        """Test default config values — no implicit LLM endpoint."""
        config = DistillerConfig()
        assert config.model is None
        assert config.api_base is None
        assert config.api_key is None
        assert config.provider is None
        assert config.temperature == 0.0
        assert not config.is_llm_configured()

    def test_config_custom_values(self):
        """Test custom config values."""
        config = DistillerConfig(
            model="gpt-4",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            provider="openai",
            temperature=0.5,
        )
        assert config.model == "gpt-4"
        assert config.api_base == "https://api.openai.com/v1"
        assert config.temperature == 0.5
        assert config.is_llm_configured()

    def test_blank_llm_fields_become_none(self):
        config = DistillerConfig(model="  ", api_base="", provider="openai")
        assert config.model is None
        assert config.api_base is None
        assert config.provider == "openai"
        assert not config.is_llm_configured()

    def test_config_model_dump(self):
        """Test config serialization."""
        config = DistillerConfig(model="test-model", temperature=0.7)
        dump = config.model_dump()
        assert dump["model"] == "test-model"
        assert dump["temperature"] == 0.7


class TestDistiller:
    """Tests for Distiller class."""

    def test_distiller_init(self):
        """Test Distiller initialization."""
        config = DistillerConfig()
        distiller = Distiller(config)
        assert distiller.config == config

    @pytest.mark.asyncio
    async def test_summarize_skips_llm_when_unconfigured(self):
        """Offline mode: no LiteLLM call without explicit LLM config."""
        config = DistillerConfig()
        distiller = Distiller(config)

        with patch("code_rag.intelligence.distiller.litellm.acompletion") as mock_comp:
            result = await distiller.summarize("def test(): pass", "test_func")
            assert result == ""
            mock_comp.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_success(self):
        """Test successful summarization."""
        config = DistillerConfig(
            model="gpt-4",
            provider="openai",
            api_base="https://api.openai.com/v1",
        )
        distiller = Distiller(config)

        with patch("code_rag.intelligence.distiller.litellm.acompletion") as mock_comp:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Test summary."))
            ]
            mock_comp.return_value = mock_response

            result = await distiller.summarize("def test(): pass", "test_func")
            assert result == "Test summary."
            mock_comp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summarize_with_ollama_provider(self):
        """Test summarization with Ollama provider."""
        config = DistillerConfig(
            model="llama3",
            provider="ollama",
            api_base="http://localhost:11434",
        )
        distiller = Distiller(config)

        with patch("code_rag.intelligence.distiller.litellm.acompletion") as mock_comp:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Ollama summary."))
            ]
            mock_comp.return_value = mock_response

            await distiller.summarize("def foo(): pass", "foo")
            call_kwargs = mock_comp.call_args.kwargs
            assert call_kwargs["model"] == "ollama/llama3"
