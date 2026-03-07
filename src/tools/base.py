from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> str:
        pass

    @property
    @abstractmethod
    def schema(self) -> dict:
        """Return the tool schema for Claude's tool_use API."""
        pass
