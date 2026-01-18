from typing import Dict, Any, Optional
from llm.providers.config import BaseLLMConfig

class GeminiConfig(BaseLLMConfig):
    safety_settings: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
