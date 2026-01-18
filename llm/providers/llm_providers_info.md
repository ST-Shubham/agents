# LLM Provider System

## Overview

The LLM Provider system is designed to provide a unified interface for interacting with different Large Language Models (LLMs) such as OpenAI, Google Gemini, and Anthropic Claude. It abstracts away the differences between provider APIs, allowing for easy switching and consistent usage across the application.

## Core Components

### 1. `LLMProvider` (Abstract Base Class)
The core of the system is the `LLMProvider` class defined in `llm/providers/provider.py`. It establishes the contract that all specific provider implementations must follow.

**Key Features:**
*   **History Management**: Maintains a list of `Message` objects (user, assistant, tool).
*   **System Message**: Handles system instructions.
*   **Abstract Interface**: Enforces implementation of generation methods.

```python
from abc import ABC, abstractmethod
from typing import List, Literal
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str

class LLMProvider(ABC):
    def __init__(self, config: BaseLLMConfig, system_message: str):
        self.config = config
        self.system_message = system_message
        self.history: List[Message] = []

    def add_user_message(self, content: str):
        # Adds user message to history
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        # Generate a response
        pass
```

### 2. `BaseLLMConfig`
Configuration is handled via Pydantic models to ensure type safety and validation. `BaseLLMConfig` provides common fields like `model`, `temperature`, and `max_tokens`.

```python
from pydantic import BaseModel, ConfigDict
from typing import Optional

class BaseLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

## Factory Pattern (`create_instance.py`)

The system uses a factory pattern to instantiate the correct provider and ensure the configuration matches the provider type.

```python
def create_provider_instance(
    provider: LLMProviderOptions,
    config: BaseLLMConfig,
    system_message: str,
) -> LLMProvider:
    # Logic to select provider class and validate config
    # Returns an instance of the specific LLMProvider
```

## How It Works

1.  **Selection**: The user or application selects a provider (e.g., `LLMProviderOptions.ANTHROPIC`).
2.  **Configuration**: A specific config object is created (e.g., `ClaudeConfig`).
3.  **Instantiation**: `create_provider_instance` is called. It verifies the config type matches the provider and initializes the provider with the system message.
4.  **Interaction**:
    *   `provider.generate("Hello")` adds the user prompt to history, calls the API, adds the response to history, and returns the text.
    *   `provider.generate_stream(...)` allows for streaming responses.
    *   `provider.generate_structured(...)` forces the output into a specific schema.

## Example Implementation

Here is how a specific provider (e.g., `ClaudeProvider`) implements the base class:

```python
class ClaudeProvider(LLMProvider):
    config_schema = ClaudeConfig

    def generate(self, prompt: str, **kwargs) -> str:
        self.add_user_message(prompt)
        # Call Anthropic API here
        response = "..." 
        self.add_assistant_message(response)
        return response
```
