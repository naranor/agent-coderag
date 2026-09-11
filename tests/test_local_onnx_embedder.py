import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import logging

from code_rag.core.constants import LOCAL_EMBEDDING_MODEL_ID
from code_rag.core.exceptions import IntelligenceError
from code_rag.core.interfaces import IEmbedder
from code_rag.intelligence.embedder import (
    Embedder,
    LocalOnnxEmbedder,
    l2_normalize_rows,
)


def _ready_session(dim: int = 384):
    session = MagicMock()
    mock_output = np.random.rand(1, 3, dim).astype(np.float32)
    session.run.return_value = [mock_output]
    mock_input = MagicMock()
    mock_input.name = "input_ids"
    session.get_inputs.return_value = [mock_input]
    out = MagicMock()
    out.shape = (None, None, dim)
    session.get_outputs.return_value = [out]
    return session


def _ready_tokenizer():
    tokenizer = MagicMock()
    encoding = MagicMock()
    encoding.ids = [1, 2, 3]
    encoding.attention_mask = [1, 1, 1]
    tokenizer.encode_batch.return_value = [encoding]
    return tokenizer


def test_embedder_alias_and_interface():
    assert Embedder is LocalOnnxEmbedder
    assert issubclass(LocalOnnxEmbedder, IEmbedder)


def test_l2_normalize_rows_unit_norm():
    matrix = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize_rows(matrix)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_ctor_binds_dimension_from_onnx_and_model_id():
    with patch(
        "tokenizers.Tokenizer.from_file", return_value=_ready_tokenizer()
    ), patch("onnxruntime.InferenceSession", return_value=_ready_session(384)), patch(
        "os.path.exists", return_value=True
    ):
        embedder = LocalOnnxEmbedder(model_path="fake/dir/model.onnx")
    assert embedder.dimension == 384
    assert embedder.model_id == LOCAL_EMBEDDING_MODEL_ID
    embedder.bind_dimension(384)


def test_bind_dimension_disagrees():
    with patch(
        "tokenizers.Tokenizer.from_file", return_value=_ready_tokenizer()
    ), patch("onnxruntime.InferenceSession", return_value=_ready_session(384)), patch(
        "os.path.exists", return_value=True
    ):
        embedder = LocalOnnxEmbedder(model_path="fake/dir/model.onnx")
    with pytest.raises(IntelligenceError, match="already bound"):
        embedder.bind_dimension(8)


def test_unbound_dimension_raises():
    with patch("os.path.exists", return_value=False):
        embedder = LocalOnnxEmbedder(model_path=None)
    with pytest.raises(IntelligenceError, match="not bound"):
        _ = embedder.dimension


@pytest.mark.asyncio
async def test_aembed_tolist_l2_and_close():
    session = _ready_session(384)
    tokenizer = _ready_tokenizer()
    enc1 = MagicMock()
    enc1.ids = [1, 2, 3]
    enc1.attention_mask = [1, 1, 1]
    enc2 = MagicMock()
    enc2.ids = [4, 5, 6]
    enc2.attention_mask = [1, 1, 1]
    tokenizer.encode_batch.return_value = [enc1, enc2]
    session.run.return_value = [np.random.rand(2, 3, 384).astype(np.float32)]

    with patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), patch(
        "onnxruntime.InferenceSession", return_value=session
    ), patch("os.path.exists", return_value=True):
        embedder = LocalOnnxEmbedder(model_path="fake/dir/model.onnx")
        rows = await embedder.aembed(["hello", "world"])
        assert isinstance(rows, list)
        assert len(rows) == 2
        assert len(rows[0]) == 384
        norms = np.linalg.norm(np.array(rows), axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-4)
        await embedder.close()
        assert embedder.session is None


@pytest.mark.asyncio
async def test_aembed_without_session_mentions_setup():
    with patch("os.path.exists", return_value=False):
        embedder = LocalOnnxEmbedder(model_path=None)
    with pytest.raises(IntelligenceError, match="setup"):
        await embedder.aembed(["x"])


def test_bind_unbound_then_reject_invalid():
    with patch("os.path.exists", return_value=False):
        embedder = LocalOnnxEmbedder(model_path=None)
    embedder.bind_dimension(8)
    assert embedder.dimension == 8
    with pytest.raises(IntelligenceError, match="positive int"):
        embedder.bind_dimension(True)  # type: ignore[arg-type]


def test_init_helpers_noop_without_model_path():
    with patch("os.path.exists", return_value=False):
        embedder = LocalOnnxEmbedder(model_path=None)
    embedder._init_tokenizer()
    embedder._init_session()
    assert embedder.tokenizer is None
    assert embedder.session is None


def test_tokenizer_falls_back_to_parent_dir(tmp_path):
    model_dir = tmp_path / "models" / "mini-lm"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"onnx")
    (tmp_path / "models" / "tokenizer.json").write_text("{}", encoding="utf-8")

    tokenizer = _ready_tokenizer()
    session = _ready_session(8)
    with patch(
        "tokenizers.Tokenizer.from_file", return_value=tokenizer
    ) as mock_tok, patch("onnxruntime.InferenceSession", return_value=session):
        embedder = LocalOnnxEmbedder(model_path=str(model_path))
    mock_tok.assert_called_once_with(str(tmp_path / "models" / "tokenizer.json"))
    assert embedder.session is session


def test_tokenizer_load_failure(tmp_path):
    model_dir = tmp_path / "mini-lm"
    model_dir.mkdir()
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"onnx")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    with patch(
        "tokenizers.Tokenizer.from_file", side_effect=ValueError("bad tokenizer")
    ):
        with pytest.raises(IntelligenceError, match="Failed to load tokenizer"):
            LocalOnnxEmbedder(model_path=str(model_path))


def test_tokenizer_missing_near_model(tmp_path):
    model_dir = tmp_path / "mini-lm"
    model_dir.mkdir()
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"onnx")
    with pytest.raises(IntelligenceError, match="tokenizer.json not found"):
        LocalOnnxEmbedder(model_path=str(model_path))


def test_embed_includes_token_type_ids():
    session = _ready_session(8)
    inp_ids = MagicMock()
    inp_ids.name = "input_ids"
    tok_types = MagicMock()
    tok_types.name = "token_type_ids"
    session.get_inputs.return_value = [inp_ids, tok_types]
    session.run.return_value = [np.random.rand(1, 3, 8).astype(np.float32)]
    tokenizer = _ready_tokenizer()
    tokenizer.encode_batch.return_value[0].type_ids = [0, 0, 0]
    with patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), patch(
        "onnxruntime.InferenceSession", return_value=session
    ), patch("os.path.exists", return_value=True):
        embedder = LocalOnnxEmbedder(model_path="fake/dir/model.onnx")
        embedder.embed(["hello"])
    kwargs = session.run.call_args.args[1]
    assert "token_type_ids" in kwargs


def test_embed_reraises_intelligence_error():
    session = _ready_session(8)
    tokenizer = _ready_tokenizer()
    with patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), patch(
        "onnxruntime.InferenceSession", return_value=session
    ), patch("os.path.exists", return_value=True):
        embedder = LocalOnnxEmbedder(model_path="fake/dir/model.onnx")
    session.run.side_effect = IntelligenceError("inner")
    with pytest.raises(IntelligenceError, match="inner"):
        embedder.embed(["hello"])


@pytest.mark.asyncio
async def test_close_without_session_is_noop():
    with patch("os.path.exists", return_value=False):
        embedder = LocalOnnxEmbedder(model_path=None)
    await embedder.close()
    assert embedder.session is None


def test_missing_global_model_warns(tmp_path, caplog):
    model_dir = tmp_path / "models" / "mini-lm"
    with patch(
        "code_rag.intelligence.embedder.get_default_model_dir", return_value=model_dir
    ):
        with caplog.at_level(logging.WARNING):
            embedder = LocalOnnxEmbedder(model_path=None)
    assert embedder.session is None
    assert "No model found" in caplog.text


def test_uses_global_model_when_present(tmp_path):
    model_dir = tmp_path / "mini-lm"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"onnx")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizer = _ready_tokenizer()
    session = _ready_session(8)
    with patch(
        "code_rag.intelligence.embedder.get_default_model_dir", return_value=model_dir
    ), patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), patch(
        "onnxruntime.InferenceSession", return_value=session
    ):
        embedder = LocalOnnxEmbedder(model_path=None)
    assert embedder.model_path == str(model_dir / "model.onnx")
    assert embedder.session is session
