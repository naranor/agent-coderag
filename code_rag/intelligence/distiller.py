import logging
import json
import importlib
from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator
from ..core.exceptions import IntelligenceError
from ..core.interfaces import IIntelligence
from ..core.constants import LLM_REQUEST_TIMEOUT
from .embedder import get_global_dir

logger = logging.getLogger(__name__)


class DistillerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.0
    embedding_base: Optional[str] = None
    embedding_key: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    relative_paths: bool = False

    @field_validator(
        "model",
        "api_base",
        "api_key",
        "provider",
        "embedding_base",
        "embedding_key",
        "embedding_model",
        "embedding_provider",
        mode="before",
    )
    @classmethod
    def _blank_string_to_none(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    def is_llm_configured(self) -> bool:
        """True when distill LLM endpoint is explicitly configured (offline otherwise)."""
        return bool(self.model and self.api_base and self.provider)

    @classmethod
    def load(cls) -> "DistillerConfig":
        """Loads config from the global agent-coderag directory."""
        config_path = get_global_dir() / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return cls(**data)
            except Exception as e:
                logger.error("Failed to load config from %s: %s", config_path, e)
        return cls()

    def save(self):
        """Saves current config to the global agent-coderag directory."""
        base_set = self.embedding_base is not None
        model_set = self.embedding_model is not None
        if base_set != model_set:
            raise IntelligenceError(
                "embedding_base and embedding_model must both be set or both unset"
            )
        config_path = get_global_dir() / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.model_dump(), f, indent=4)
            logger.info("Config saved to %s", config_path)
        except Exception as e:
            logger.error("Failed to save config to %s: %s", config_path, e)


class Distiller(IIntelligence):
    """
    LLM-based code analyst that extracts the 'intent' from raw code.
    """

    def __init__(self, config: DistillerConfig):
        self.config = config

    async def summarize(self, code: str, unit_name: str) -> str:
        """
        Generates a concise technical summary of what the code DOES.

        When LLM is not configured, returns "" without calling the network
        (offline / signature-fallback mode).
        """
        if not self.config.is_llm_configured():
            logger.debug("LLM not configured; skipping distillation for %s", unit_name)
            return ""

        prompt = f"""
Analyze the following code block for '{unit_name}'.
Provide a concise, 1-2 sentence technical description of its core logic and intent.
Focus on WHAT it accomplishes and its role in the system.
DO NOT repeat the signature.
DO NOT include docstrings or comments in your summary.

CODE:
{code}

SUMMARY:
"""
        model_id = str(self.config.model)
        if self.config.provider == "ollama" and not model_id.startswith("ollama/"):
            model_id = f"ollama/{model_id}"

        litellm = importlib.import_module("litellm")
        response = await litellm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            timeout=LLM_REQUEST_TIMEOUT,
            custom_llm_provider=self.config.provider,
        )

        summary = response.choices[0].message.content.strip()
        return summary
