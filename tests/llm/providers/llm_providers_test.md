# LLM Providers Testing Strategy

## Overview

This directory (`tests/llm/providers/`) contains tests for the LLM Provider system, ensuring that the unified interface, factory logic, and specific provider implementations function as expected.

## Test Files and Scope

### 1. Base Provider Logic (`tests/llm/providers/provider.py`)

This file tests the abstract base class `LLMProvider` by creating a `MockProvider`.

- **Goal**: Ensure shared logic works correctly without relying on a real external API.
- **Key Tests**:
  - `test_base_provider_history`: Verifies that `add_user_message` and `add_assistant_message` correctly append `Message` objects to the `history` list.
  - `test_base_provider_payload_build`: Verifies that `_build_messages_payload` correctly formats the system message and conversation history into a list of dictionaries.

### 2. Factory Pattern & Type Safety (`tests/llm/providers/create_instance.py`)

This file tests the `create_provider_instance` factory function.

- **Goal**: Ensure the correct provider class is instantiated for a given option and that configuration types are enforced.
- **Key Tests**:
  - `test_create_openai_instance`, `test_create_gemini_instance`, `test_create_claude_instance`: Verify that passing the corresponding `LLMProviderOptions` returns an instance of `OpenAIProvider`, `GeminiProvider`, or `ClaudeProvider`, respectively.
  - `test_create_instance_invalid_config`: It verifies that a `TypeError` is raised if a mismatch occurs (e.g., passing `OpenAIConfig` when requesting the `GEMINI` provider).
  - `test_llm_provider_type_resolver`: Verifies internal mapping logic between enums, provider classes, and config classes.

### 3. Interface Verification (Mock Provider)

This section defines the core interface behaviors and shared logic validated using a `MockProvider`. Specific provider validations (Claude, OpenAI, Gemini) are described in their respective documentation files.

- **Goal**: Ensure the `LLMProvider` base class and its shared utilities function correctly and consistently across all implementations.
- **Key Validations**:
  - **Message History**: Verifying that `add_user_message` and `add_assistant_message` correctly append `Message` objects.
  - **Payload Construction**: Ensuring `_build_messages_payload` correctly formats the system message and conversation history.
  - **State Management**: Confirming that history and configuration are correctly maintained within the provider instance.

## Summary

The testing strategy covers:

1.  **Abstract Logic**: Via `MockProvider`.
2.  **Safety**: Via factory type checks.
3.  **Concrete Behavior**: Via specific provider tests.
