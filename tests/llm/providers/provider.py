from llm.providers.provider import LLMProvider, Message
from llm.providers.config import BaseLLMConfig
from typing import Generator, Dict, Any, List
from pydantic import BaseModel

class MockConfig(BaseLLMConfig):
    pass

class MockProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return "mock"
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield "mock"
    def generate_structured(self, prompt: str, response_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {}
    def generate_with_tools(self, prompt: str, tools: List[BaseModel], **kwargs):
        return {}

def test_base_provider_history():
    config = MockConfig(model="test")
    provider = MockProvider(config=config, system_message="System")
    
    provider.add_user_message("User")
    provider.add_assistant_message("Assistant")
    
    assert len(provider.history) == 2
    assert provider.history[0] == Message(role="user", content="User")
    assert provider.history[1] == Message(role="assistant", content="Assistant")

def test_base_provider_payload_build():
    config = MockConfig(model="test")
    provider = MockProvider(config=config, system_message="System")
    provider.add_user_message("Hello")
    
    # Mocking _system_message which is what _build_messages_payload uses
    provider._system_message = "System" 
    payload = provider._build_messages_payload()
    
    assert len(payload) == 2
    assert payload[0] == {"role": "system", "content": "System"}
    assert payload[1] == {"role": "user", "content": "Hello"}
