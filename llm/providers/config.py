from pydantic import BaseModel, ConfigDict
from typing import Optional

class BaseLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None