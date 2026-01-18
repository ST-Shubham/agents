from enum import Enum

# Define Enum for LLM Provider Options
class LLMProviderOptions(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"