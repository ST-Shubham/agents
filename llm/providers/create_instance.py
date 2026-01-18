from typing import Type, Union

from llm.providers.claude.config import ClaudeConfig
from llm.providers.claude.provider import ClaudeProvider
from llm.providers.config import BaseLLMConfig
from llm.providers.provider_options import LLMProviderOptions
from llm.providers.provider import LLMProvider
from llm.providers.openai.provider import OpenAIProvider
from llm.providers.openai.config import OpenAIConfig
from llm.providers.gemini.provider import GeminiProvider
from llm.providers.gemini.config import GeminiConfig

class LLMProviderType:
    provider_cls: Type[LLMProvider]
    config_cls: Type[BaseLLMConfig]
    
    def __init__(self, provider_option: LLMProviderOptions):
        match provider_option:
            case LLMProviderOptions.OPENAI:
                self.provider_cls = OpenAIProvider
                self.config_cls = OpenAIConfig
            case LLMProviderOptions.GEMINI:
                self.provider_cls = GeminiProvider
                self.config_cls = GeminiConfig
            case LLMProviderOptions.ANTHROPIC:
                self.provider_cls = ClaudeProvider
                self.config_cls = ClaudeConfig
            case _:
                raise ValueError(f"Provider {provider_option} is not supported.")

def create_provider_instance(
    provider: LLMProviderOptions,
    config: BaseLLMConfig,
    system_message: str,
) -> LLMProvider:
    
    provider_type = LLMProviderType(provider)

    # Runtime safety: ensure correct config type
    if not isinstance(config, provider_type.config_cls):
        raise TypeError(
            f"{provider.value} expects {provider_type.config_cls.__name__}"
        )

    return provider_type.provider_cls(config=config, system_message=system_message)