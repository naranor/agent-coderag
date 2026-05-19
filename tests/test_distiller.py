import pytest
from unittest.mock import MagicMock, patch

from code_rag.intelligence.distiller import Distiller, DistillerConfig


class TestDistillerConfig:
    """Tests for DistillerConfig."""

    def test_load_default_config(self):
        """Test loading default config."""
        config = DistillerConfig.load()
        assert config.model == "auto"
        assert config.provider == "openai"

    def test_config_defaults(self):
        """Test default config values."""
        config = DistillerConfig()
        assert config.model == "auto"
        assert config.api_base == "http://localhost:8081/api/v1"
        assert config.api_key is None
        assert config.provider == "openai"
        assert config.temperature == 0.0

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
    async def test_summarize_success(self):
        """Test successful summarization."""
        config = DistillerConfig(model="gpt-4", provider="openai")
        distiller = Distiller(config)

        with patch("code_rag.intelligence.distiller.litellm.acompletion") as mock_comp:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Test summary."))
            ]
            mock_comp.return_value = mock_response

            result = await distiller.summarize("def test(): pass", "test_func")
            assert result == "Test summary."

    @pytest.mark.asyncio
    async def test_summarize_with_ollama_provider(self):
        """Test summarization with Ollama provider."""
        config = DistillerConfig(model="llama3", provider="ollama")
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

    @pytest.mark.asyncio
    async def test_summarize_empty_code(self):
        """Test summarization with empty code."""
        config = DistillerConfig()
        distiller = Distiller(config)

        with patch("code_rag.intelligence.distiller.litellm.acompletion") as mock_comp:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Empty summary."))
            ]
            mock_comp.return_value = mock_response

            await distiller.summarize("", "empty_func")
            assert "empty_func" in mock_comp.call_args.kwargs["messages"][0]["content"]
