"""Structured outputs used inside the benchmark synthesis pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DraftMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    is_test_turn: bool = False


class SingleTurnDraft(BaseModel):
    input: str = Field(..., min_length=1)


class ConversationDraft(BaseModel):
    conversation: list[DraftMessage] = Field(..., min_length=1)
    memory_planted: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conversation(self) -> "ConversationDraft":
        test_turns = [msg for msg in self.conversation if msg.is_test_turn]
        if len(test_turns) != 1:
            raise ValueError("conversation must contain exactly one test turn")
        if self.conversation[-1].role != "user":
            raise ValueError("conversation must end with a user message")
        if not self.conversation[-1].is_test_turn:
            raise ValueError("final message must be marked as is_test_turn")

        for idx, msg in enumerate(self.conversation):
            expected = "user" if idx % 2 == 0 else "assistant"
            if msg.role != expected:
                raise ValueError("conversation roles must alternate user/assistant")
        return self

    @property
    def test_input(self) -> str:
        return self.conversation[-1].content

    @property
    def history(self) -> list[dict[str, str]]:
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation[:-1]
        ]
