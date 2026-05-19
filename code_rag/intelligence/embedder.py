import os
import logging
import numpy as np
from typing import List, Optional
from pathlib import Path
import onnxruntime as ort
from tokenizers import Tokenizer
from ..core.constants import MAX_TOKEN_LENGTH, PAD_ID, PAD_TOKEN
from ..core.exceptions import IntelligenceError

logger = logging.getLogger(__name__)


def get_global_dir() -> Path:
    """Returns the default global directory for agent-coderag data (cross-platform)."""
    if os.name == "nt":  # Windows
        base_dir = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
    else:  # Linux/macOS
        base_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base_dir / "agent-coderag"


def get_default_model_dir() -> Path:
    """Returns the default global directory for models."""
    return get_global_dir() / "models" / "mini-lm"


class Embedder:
    """
    Local multilingual embedder using ONNX Runtime and Tokenizers.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[Tokenizer] = None

        if not self.model_path:
            global_dir = get_default_model_dir()
            potential_path = global_dir / "model.onnx"
            if potential_path.exists():
                self.model_path = str(potential_path)
                logger.info("Using global model from %s", self.model_path)
            else:
                logger.warning(
                    "No model found at %s. Please run 'agent-coderag setup'.",
                    potential_path,
                )

        if self.model_path and os.path.exists(self.model_path):
            self._init_tokenizer()
            self._init_session()

    def _init_tokenizer(self):
        if not self.model_path:
            return

        model_dir = os.path.dirname(self.model_path)
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            tokenizer_file = os.path.join(os.path.dirname(model_dir), "tokenizer.json")

        if os.path.exists(tokenizer_file):
            try:
                self.tokenizer = Tokenizer.from_file(tokenizer_file)
                self.tokenizer.enable_padding(pad_id=PAD_ID, pad_token=PAD_TOKEN)  # nosec B106
                self.tokenizer.enable_truncation(max_length=MAX_TOKEN_LENGTH)
                logger.debug("Loaded tokenizer from %s", tokenizer_file)
            except Exception as e:
                logger.error("Failed to load tokenizer from %s: %s", tokenizer_file, e)
                raise IntelligenceError(f"Failed to load tokenizer: {e}") from e
        else:
            err_msg = f"tokenizer.json not found near {self.model_path}"
            logger.error(err_msg)
            raise IntelligenceError(err_msg)

    def _init_session(self):
        if not self.model_path:
            return

        try:
            self.session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            logger.info("ONNX session initialized with model: %s", self.model_path)
        except Exception as e:
            logger.error("Failed to initialize ONNX session: %s", e)
            raise IntelligenceError(f"Failed to initialize ONNX session: {e}") from e

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.session or not self.tokenizer:
            raise IntelligenceError(
                "Embedder not initialized. Please ensure models are downloaded by running 'agent-coderag setup'."
            )

        try:
            encodings = self.tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            model_inputs = [i.name for i in self.session.get_inputs()]
            if "token_type_ids" in model_inputs:
                inputs["token_type_ids"] = np.array(
                    [e.type_ids for e in encodings], dtype=np.int64
                )

            outputs = self.session.run(None, inputs)
            embeddings = self._mean_pooling(outputs[0], attention_mask)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            return embeddings / norms
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            raise IntelligenceError(f"Failed to generate embeddings: {e}") from e

    def _mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def close(self):
        """Releases the ONNX Runtime session and resources."""
        if self.session:
            self.session = None
            logger.info("Embedder resources released.")
