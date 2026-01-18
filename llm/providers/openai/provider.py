from llm.providers.provider import LLMProvider
from llm.providers.openai.config import OpenAIConfig
from typing import Generator, Dict, Any, List
from pydantic import BaseModel

class OpenAIProvider(LLMProvider):
    config_schema = OpenAIConfig

    def generate(self, prompt: str, **kwargs) -> str:
        self.add_user_message(prompt)
        # OpenAI SDK call here
        response = "openai-response"
        self.add_assistant_message(response)
        return response

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield "openai-stream"

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
