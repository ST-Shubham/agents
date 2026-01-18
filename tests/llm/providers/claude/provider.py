from llm.providers.claude.provider import ClaudeProvider
from llm.providers.claude.config import ClaudeConfig

def test_claude_provider_generate():
    config = ClaudeConfig(model="claude-3")
    provider = ClaudeProvider(config=config, system_message="Test System")
    
    response = provider.generate("Hello")
    assert response == "claude-response"
    assert len(provider.history) == 2
    assert provider.history[0].role == "user"
    assert provider.history[1].role == "assistant"

def test_claude_provider_history():
    config = ClaudeConfig(model="claude-3")
    provider = ClaudeProvider(config=config, system_message="Test System")
    
    provider.add_user_message("Message 1")
    provider.add_assistant_message("Reply 1")
    
    assert len(provider.history) == 2
    assert provider.history[0].content == "Message 1"
    assert provider.history[1].content == "Reply 1"