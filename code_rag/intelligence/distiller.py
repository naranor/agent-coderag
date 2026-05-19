import logging
import json
from typing import Optional
import litellm
from pydantic import BaseModel
from ..core.interfaces import IIntelligence
from ..core.constants import LLM_REQUEST_TIMEOUT
from .embedder import get_global_dir

logger = logging.getLogger(__name__)


class DistillerConfig(BaseModel):
    model: str = "auto"
    api_base: str = "http://localhost:8081/api/v1"
    api_key: Optional[str] = None
    provider: str = "openai"
    temperature: float = 0.0

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
        """
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
        model_id = self.config.model
        if self.config.provider == "ollama" and not model_id.startswith("ollama/"):
            model_id = f"ollama/{model_id}"

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
