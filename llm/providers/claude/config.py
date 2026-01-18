from typing import Dict, Any, Optional
from llm.providers.config import BaseLLMConfig

class ClaudeConfig(BaseLLMConfig):
    top_k: Optional[int] = None
