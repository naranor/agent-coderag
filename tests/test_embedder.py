import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch

from code_rag.intelligence.embedder import (
    Embedder,
    get_default_model_dir,
    get_global_dir,
)
from code_rag.core.exceptions import IntelligenceError


class TestEmbedder:
    """Tests for Embedder class."""

    def test_embedder_init_no_path(self):
        with patch("pathlib.Path.exists", return_value=False):
            embedder = Embedder(model_path=None)
            assert embedder.model_path is None
            assert embedder.session is None

    def test_embedder_init_invalid_path(self):
        # We patch exists to return False for the model path
        with patch("pathlib.Path.exists", return_value=False):
            embedder = Embedder(model_path="/nonexistent/path/model.onnx")
            assert embedder.session is None

    def test_embed_no_session_raises_error(self):
        with patch("pathlib.Path.exists", return_value=False):
            embedder = Embedder(model_path=None)
            with pytest.raises(IntelligenceError) as exc:
                embedder.embed(["test text", "another text"])
            assert "Embedder not initialized" in str(exc.value)

    def test_embed_single_text_raises_error(self):
        with patch("pathlib.Path.exists", return_value=False):
            embedder = Embedder(model_path=None)
            with pytest.raises(IntelligenceError) as exc:
                embedder.embed(["single text"])
            assert "Embedder not initialized" in str(exc.value)

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux-specific path handling")
    def test_get_global_dir_posix_default(self):
        """Test global directory on POSIX without XDG."""
        with patch("code_rag.intelligence.embedder.os.name", "posix"):
            with patch.dict(os.environ, {}, clear=True):
                # Path.home() may not be available in test environment
                with patch("pathlib.Path.home", return_value=Path("/home/test")):
                    result = get_global_dir()
                    assert "agent-coderag" in str(result)

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux-specific path handling")
    def test_get_global_dir_xdg(self):
        """Test global directory with XDG_CACHE_HOME."""
        with patch("code_rag.intelligence.embedder.os.name", "posix"):
            with patch.dict(os.environ, {"XDG_CACHE_HOME": "/test/xdg"}):
                with patch("pathlib.Path.home", return_value=Path("/home/test")):
                    result = get_global_dir()
                    assert "/test/xdg" in str(result)

    def test_get_default_model_dir(self):
        result = get_default_model_dir()
        assert "mini-lm" in str(result)
