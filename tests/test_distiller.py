import pytest
from unittest.mock import MagicMock, patch

from code_rag.intelligence.distiller import Distiller, DistillerConfig


class TestDistillerConfig:
    def test_load_default_config(self):
        config = DistillerConfig.load()
        assert config.model == "auto"
        assert config.provider == "openai"
    
    def test_config_defaults(self):
        config = DistillerConfig()
        assert config.model == "auto"
        assert config.temperature == 0.0
    
    def test_config_custom_values(self):
        config = DistillerConfig(model="gpt-4", provider="ollama", temperature=0.5)
        assert config.model == "gpt-4"
        assert config.temperature == 0.5


class TestDistiller:
    def test_distiller_init(self):
        config = DistillerConfig()
        distiller = Distiller(config)
        assert distiller.config == config
    
    @pytest.mark.asyncio
    async def test_summarize_success(self):
        config = DistillerConfig(model="gpt-4", provider="openai")
        distiller = Distiller(config)
        
        with patch('code_rag.intelligence.distiller.litellm.acompletion') as mock_comp:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Test summary."))]
            mock_comp.return_value = mock_response
            
            result = await distiller.summarize("def test(): pass", "test_func")
            assert result == "Test summary."