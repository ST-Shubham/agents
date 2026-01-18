from typing import List, Literal, Optional
from pydantic import BaseModel
from llm.providers.config import BaseLLMConfig
from llm.providers.provider_options import LLMProviderOptions

# Model for messages
class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    
# Base Provider class (llm-provider)
from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, List, Type

class LLMProvider(ABC):
    def __init__(self, config: BaseLLMConfig, system_message: str):
        self.config = config
        self.system_message = system_message
        self.history: List[Message] = []

    # -------------------------
    # History Management
    # -------------------------
    def add_user_message(self, content: str):
        self.history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str):
        self.history.append(Message(role="assistant", content=content))

	# Todo: Implement a good rollback and fine tuned selection system or focus on chats between a and b
    # def rollback(self, steps: int = 1):
    #     """Go back N messages (excluding system)."""
    #     if steps > 0:
    #         self.history = self.history[:-steps]

    def save_history(self, identifier: str) -> List[Message]:
        # Todo: Save history in db
        return self.history.copy()

    def compress_history(self, summarizer: "LLMProvider"):
        """Replace history with a summary."""
        summary = summarizer.generate(
            "Summarize the following conversation:\n"
            + "\n\n".join((m.role + ": " + m.content) for m in self.history)
        )
        self.history = [
            Message(role="assistant", content=summary)
        ]
    
    # -------------------------
    # Messages Management
    # -------------------------
    
    # allow changing system message at runtime (still not persisted)
    def set_system_message(self, msg: Optional[str]):
        self._system_message = msg

    # helpers to build provider payload (default behavior)
    def _build_messages_payload(self) -> List[Dict[str, str]]:
        """
        Default neutral representation: system injected as first message if present.
        Child providers may override to fit their API.
        """
        msgs = []
        if self._system_message:
            msgs.append({"role": "system", "content": self._system_message})
        for m in self.history:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    # -------------------------
    # Generation APIs
    # -------------------------
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[BaseModel],
        enforce_tool_use: bool = False,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        pass