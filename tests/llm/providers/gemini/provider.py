from llm.providers.gemini.provider import GeminiProvider
from llm.providers.gemini.config import GeminiConfig

def test_gemini_provider_generate():
    config = GeminiConfig(model="gemini-1.5-flash")
    provider = GeminiProvider(config=config, system_message="Test System")
    
    response = provider.generate("Hello")
    assert response == "gemini-response"
    assert len(provider.history) == 2
    assert provider.history[0].role == "user"
    assert provider.history[1].role == "assistant"

def test_gemini_provider_history():
    config = GeminiConfig(model="gemini-1.5-flash")
    provider = GeminiProvider(config=config, system_message="Test System")
    
    provider.add_user_message("Message 1")
    provider.add_assistant_message("Reply 1")
    
    assert len(provider.history) == 2
    assert provider.history[0].content == "Message 1"
    assert provider.history[1].content == "Reply 1"
