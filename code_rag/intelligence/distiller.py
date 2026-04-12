import logging
import litellm
from pydantic import BaseModel, Field
from typing import Optional
from ..core.interfaces import IIntelligence
from ..core.models import KnowledgeUnit

logger = logging.getLogger(__name__)

class DistillerConfig(BaseModel):
    model: str = "auto"
    api_base: Optional[str] = "http://192.168.92.2:8383/api/v1"
    api_key: Optional[str] = "sk-placeholder"
    provider: str = "openai" # "openai" or "ollama"
    temperature: float = 0.0

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
        try:
            # Prepare model identifier for LiteLLM
            # For Ollama, it usually looks like "ollama/llama3"
            model_id = self.config.model
            if self.config.provider == "ollama" and not model_id.startswith("ollama/"):
                model_id = f"ollama/{model_id}"

            response = await litellm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.config.api_base,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                timeout=30,
                custom_llm_provider="openai"
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logger.error(f"Distillation failed for {unit_name}: {e}")
            return f"Error during distillation: {str(e)}"
