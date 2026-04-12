import os
import logging
import numpy as np
from typing import List
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class Embedder:
    """
    Local multilingual embedder using ONNX Runtime.
    Default model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    """
    
    def __init__(self, model_path: str = None, tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_path = model_path
        self.session = None
        
        # v5.44: Prefer local tokenizer if model_path is provided
        if self.model_path:
            model_dir = os.path.dirname(os.path.dirname(self.model_path)) # Go up from onnx/model.onnx
            if os.path.exists(os.path.join(model_dir, "tokenizer.json")):
                tokenizer_name = model_dir
                logger.info(f"Loading local tokenizer from {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.model_path and os.path.exists(self.model_path):
            self._init_session()

    def _init_session(self):
        """Initializes ONNX session."""
        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            logger.info(f"ONNX session initialized with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts."""
        if not self.session:
            # Fallback to a placeholder if model is not loaded yet
            # In production, this should trigger model download/loading
            logger.warning("Embedder session not initialized. Returning zero vectors.")
            return np.zeros((len(texts), 384), dtype=np.float32)

        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        
        # Get inputs for the model
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        outputs = self.session.run(None, inputs)
        
        # Apply mean pooling
        embeddings = self._mean_pooling(outputs[0], encoded["attention_mask"])
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def _mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
