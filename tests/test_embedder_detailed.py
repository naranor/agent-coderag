import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from code_rag.intelligence.embedder import Embedder
from code_rag.core.exceptions import IntelligenceError


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    # Mock encoding result
    encoding = MagicMock()
    encoding.ids = [1, 2, 3]
    encoding.attention_mask = [1, 1, 1]
    tokenizer.encode_batch.return_value = [encoding]
    return tokenizer


@pytest.fixture
def mock_session():
    session = MagicMock()
    # Mock ONNX output: [batch, seq_len, dim]
    mock_output = np.random.rand(1, 3, 384).astype(np.float32)
    session.run.return_value = [mock_output]
    # Required for the code that checks session inputs
    mock_input = MagicMock()
    mock_input.name = "input_ids"
    session.get_inputs.return_value = [mock_input]
    return session


class TestEmbedderDetailed:
    """Detailed tests for Embedder including mean pooling logic."""

    def test_embedder_init_success(self, mock_tokenizer, mock_session):
        with patch(
            "tokenizers.Tokenizer.from_file", return_value=mock_tokenizer
        ), patch("onnxruntime.InferenceSession", return_value=mock_session), patch(
            "os.path.exists", return_value=True
        ):
            embedder = Embedder(model_path="fake/dir/model.onnx")
            assert embedder.session is not None
            assert embedder.tokenizer is not None

    def test_embedder_init_session_failure(self, mock_tokenizer):
        with patch(
            "tokenizers.Tokenizer.from_file", return_value=mock_tokenizer
        ), patch(
            "onnxruntime.InferenceSession", side_effect=RuntimeError("GPU Error")
        ), patch("os.path.exists", return_value=True):
            # Our code re-raises it as IntelligenceError
            with pytest.raises(IntelligenceError):
                Embedder(model_path="fake/dir/model.onnx")

    def test_embed_logic(self, mock_tokenizer, mock_session):
        with patch(
            "tokenizers.Tokenizer.from_file", return_value=mock_tokenizer
        ), patch("onnxruntime.InferenceSession", return_value=mock_session), patch(
            "os.path.exists", return_value=True
        ):
            embedder = Embedder(model_path="fake/dir/model.onnx")
            texts = ["hello", "world"]

            # Use uniform lengths to avoid numpy inhomogeneous array error in test
            enc1 = MagicMock()
            enc1.ids = [1, 2, 3]
            enc1.attention_mask = [1, 1, 1]
            enc2 = MagicMock()
            enc2.ids = [4, 5, 6]
            enc2.attention_mask = [1, 1, 1]
            mock_tokenizer.encode_batch.return_value = [enc1, enc2]

            # Mock output for 2 texts
            mock_output = np.random.rand(2, 3, 384).astype(np.float32)
            mock_session.run.return_value = [mock_output]

            embeddings = embedder.embed(texts)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (2, 384)

    def test_embedder_close(self, mock_tokenizer, mock_session):
        with patch(
            "tokenizers.Tokenizer.from_file", return_value=mock_tokenizer
        ), patch("onnxruntime.InferenceSession", return_value=mock_session), patch(
            "os.path.exists", return_value=True
        ):
            embedder = Embedder(model_path="fake/dir/model.onnx")
            embedder.close()
            assert embedder.session is None

    def test_embed_empty_list_triggers_error(self, mock_tokenizer, mock_session):
        with patch(
            "tokenizers.Tokenizer.from_file", return_value=mock_tokenizer
        ), patch("onnxruntime.InferenceSession", return_value=mock_session), patch(
            "os.path.exists", return_value=True
        ):
            embedder = Embedder(model_path="fake/dir/model.onnx")
            # If tokenizer returns empty list, it might fail later.
            mock_tokenizer.encode_batch.return_value = []

            with pytest.raises(IntelligenceError):
                embedder.embed([])

    def test_embed_not_initialized_error(self):
        # Create embedder without session
        with patch("os.path.exists", return_value=False):
            embedder = Embedder()
            with pytest.raises(IntelligenceError) as exc:
                embedder.embed(["test"])
            assert "Embedder not initialized" in str(exc.value)
