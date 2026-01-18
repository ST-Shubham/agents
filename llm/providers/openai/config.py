from typing import Optional
from llm.providers.config import BaseLLMConfig

class OpenAIConfig(BaseLLMConfig):
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
