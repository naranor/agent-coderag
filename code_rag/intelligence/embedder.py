import os
import logging
import numpy as np
from typing import List
from pathlib import Path
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

def get_global_dir() -> Path:
    """Returns the default global directory for code-rag data (cross-platform)."""
    if os.name == 'nt': # Windows
        base_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else: # Linux/macOS
        base_dir = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    
    return base_dir / "code-rag"

def get_default_model_dir() -> Path:
    """Returns the default global directory for models."""
    return get_global_dir() / "models" / "mini-lm"

class Embedder:
    """
    Local multilingual embedder using ONNX Runtime and Tokenizers.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.session = None
        self.tokenizer = None
        
        if not self.model_path:
            global_dir = get_default_model_dir()
            potential_path = global_dir / "model.onnx"
            if potential_path.exists():
                self.model_path = str(potential_path)
                logger.info(f"Using global model from {self.model_path}")
            else:
                logger.warning(f"No model found at {potential_path}. Please run 'code-rag setup'.")
        
        if self.model_path and os.path.exists(self.model_path):
            self._init_tokenizer()
            self._init_session()

    def _init_tokenizer(self):
        model_dir = os.path.dirname(self.model_path)
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            tokenizer_file = os.path.join(os.path.dirname(model_dir), "tokenizer.json")
            
        if os.path.exists(tokenizer_file):
            try:
                self.tokenizer = Tokenizer.from_file(tokenizer_file)
                self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                self.tokenizer.enable_truncation(max_length=512)
                logger.debug(f"Loaded tokenizer from {tokenizer_file}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer from {tokenizer_file}: {e}")
        else:
            logger.error(f"tokenizer.json not found near {self.model_path}")

    def _init_session(self):
        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            logger.info(f"ONNX session initialized with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.session or not self.tokenizer:
            return np.zeros((len(texts), 384), dtype=np.float32)

        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        model_inputs = [i.name for i in self.session.get_inputs()]
        if "token_type_ids" in model_inputs:
            inputs["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self.session.run(None, inputs)
        embeddings = self._mean_pooling(outputs[0], attention_mask)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms

    def _mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
