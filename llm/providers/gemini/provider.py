from llm.providers.provider import LLMProvider
from llm.providers.gemini.config import GeminiConfig
from typing import Generator, Dict, Any, List
from pydantic import BaseModel

class GeminiProvider(LLMProvider):
    config_schema = GeminiConfig

    def generate(self, prompt: str, **kwargs) -> str:
        self.add_user_message(prompt)
        response = "gemini-response"
        self.add_assistant_message(response)
        return response

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield "gemini-stream"

    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        return {}

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[BaseModel],
        enforce_tool_use: bool = False,
        stream: bool = False,
        **kwargs
    ):
        return {}
