import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch

from code_rag.intelligence.embedder import Embedder, get_default_model_dir, get_global_dir


class TestEmbedder:
    """Tests for Embedder class."""
    
    def test_embedder_init_no_path(self):
        embedder = Embedder(model_path=None)
        assert embedder.model_path is None
        assert embedder.session is None
    
    def test_embedder_init_invalid_path(self):
        embedder = Embedder(model_path="/nonexistent/path/model.onnx")
        assert embedder.session is None
    
    def test_embed_no_session_returns_zeros(self):
        embedder = Embedder(model_path=None)
        result = embedder.embed(["test text", "another text"])
        assert result.shape == (2, 384)
        assert np.all(result == 0)
    
    def test_embed_single_text(self):
        embedder = Embedder(model_path=None)
        result = embedder.embed(["single text"])
        assert result.shape == (1, 384)
    
    def test_get_global_dir_posix_default(self):
        with patch('code_rag.intelligence.embedder.os.name', 'posix'):
            with patch.dict(os.environ, {}, clear=True):
                result = get_global_dir()
                assert 'agent-coderag' in str(result)
    
    def test_get_global_dir_xdg(self):
        with patch('code_rag.intelligence.embedder.os.name', 'posix'):
            with patch.dict(os.environ, {'XDG_CACHE_HOME': '/test/xdg'}):
                result = get_global_dir()
                assert '/test/xdg' in str(result)
    
    def test_get_default_model_dir(self):
        result = get_default_model_dir()
        assert 'mini-lm' in str(result)
    
    @pytest.mark.skip(reason="Requires real ONNX model")
    @patch('code_rag.intelligence.embedder.ort.InferenceSession')
    @patch('code_rag.intelligence.embedder.Tokenizer')
    def test_embedder_with_mock_model(self, mock_tokenizer_cls, mock_session_cls):
        """Test Embedder with mock model."""
        pass