import pytest
from llm.providers.create_instance import create_provider_instance, LLMProviderType
from llm.providers.provider_options import LLMProviderOptions
from llm.providers.openai.config import OpenAIConfig
from llm.providers.openai.provider import OpenAIProvider
from llm.providers.gemini.config import GeminiConfig
from llm.providers.gemini.provider import GeminiProvider
from llm.providers.claude.config import ClaudeConfig
from llm.providers.claude.provider import ClaudeProvider

def test_create_openai_instance():
    config = OpenAIConfig(model="gpt-4")
    system_message = "You are a helpful assistant."
    provider = create_provider_instance(LLMProviderOptions.OPENAI, config, system_message)
    
    assert isinstance(provider, OpenAIProvider)
    assert provider.config == config
    assert provider.system_message == system_message

def test_create_gemini_instance():
    config = GeminiConfig(model="gemini-pro")
    system_message = "You are a helpful assistant."
    provider = create_provider_instance(LLMProviderOptions.GEMINI, config, system_message)
    
    assert isinstance(provider, GeminiProvider)
    assert provider.config == config
    assert provider.system_message == system_message

def test_create_claude_instance():
    config = ClaudeConfig(model="claude-3")
    system_message = "You are a helpful assistant."
    provider = create_provider_instance(LLMProviderOptions.ANTHROPIC, config, system_message)
    
    assert isinstance(provider, ClaudeProvider)
    assert provider.config == config
    assert provider.system_message == system_message

def test_create_instance_invalid_config():
    # Try to use OpenAIConfig with Gemini provider
    config = OpenAIConfig(model="gpt-4")
    system_message = "Test"
    
    with pytest.raises(TypeError) as excinfo:
        create_provider_instance(LLMProviderOptions.GEMINI, config, system_message)
    
    assert "gemini expects GeminiConfig" in str(excinfo.value)

    # Try to use OpenAIConfig with Claude provider
    with pytest.raises(TypeError) as excinfo:
        create_provider_instance(LLMProviderOptions.ANTHROPIC, config, system_message)
    
    assert "anthropic expects ClaudeConfig" in str(excinfo.value)

def test_llm_provider_type_resolver():
    resolver = LLMProviderType(LLMProviderOptions.OPENAI)
    assert resolver.provider_cls == OpenAIProvider
    assert resolver.config_cls == OpenAIConfig
    
    resolver = LLMProviderType(LLMProviderOptions.GEMINI)
    assert resolver.provider_cls == GeminiProvider
    assert resolver.config_cls == GeminiConfig

    resolver = LLMProviderType(LLMProviderOptions.ANTHROPIC)
    assert resolver.provider_cls == ClaudeProvider
    assert resolver.config_cls == ClaudeConfig

def test_unsupported_provider():
    # Mocking or using a value that might exist in Enum but not in resolver if added later
    # Using a fake value for testing if possible, but Enum limits this. 
    # Since all current Enums are supported, we can't easily test "unsupported" 
    # unless we mock or if there's a value not in the match case.
    # The original test assumed ANTHROPIC was unsupported, but now it is.
    pass