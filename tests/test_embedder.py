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
    
    @patch('code_rag.intelligence.embedder.ort.InferenceSession')
    @patch('code_rag.intelligence.embedder.Tokenizer')
    def test_embedder_with_mock_model(self, mock_tokenizer_cls, mock_session_cls):
        """Test Embedder with mock model."""
        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1]
        mock_encoding.type_ids = [0, 0, 0]
        mock_tokenizer.encode_batch.return_value = [mock_encoding]
        mock_tokenizer.from_file.return_value = mock_tokenizer
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        mock_output = np.random.rand(1, 12, 384).astype(np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_output]
        mock_session.get_inputs.return_value = [MagicMock(name='input_ids')]
        mock_session_cls.return_value = mock_session
        
        with patch('code_rag.intelligence.embedder.os.path.exists', return_value=True):
            embedder = Embedder(model_path="/tmp/model.onnx")
            result = embedder.embed(["test text"])
            assert result.shape == (1, 384)