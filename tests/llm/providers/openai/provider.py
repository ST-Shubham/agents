from llm.providers.openai.provider import OpenAIProvider
from llm.providers.openai.config import OpenAIConfig

def test_openai_provider_generate():
    config = OpenAIConfig(model="gpt-3.5-turbo")
    provider = OpenAIProvider(config=config, system_message="Test System")
    
    response = provider.generate("Hello")
    assert response == "openai-response"
    assert len(provider.history) == 2
    assert provider.history[0].role == "user"
    assert provider.history[1].role == "assistant"

def test_openai_provider_history():
    config = OpenAIConfig(model="gpt-3.5-turbo")
    provider = OpenAIProvider(config=config, system_message="Test System")
    
    provider.add_user_message("Message 1")
    provider.add_assistant_message("Reply 1")
    
    assert len(provider.history) == 2
    assert provider.history[0].content == "Message 1"
    assert provider.history[1].content == "Reply 1"
