from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class RewritePayload(BaseModel):
    type: Literal["rewrite", "clarify"]
    content: str = Field(..., min_length=1)


@dataclass(frozen=True)
class RewriteResult:
    type: Literal["rewrite", "clarify"]
    content: str

    @classmethod
    def parse(cls, raw: str | dict[str, Any] | None, fallback: str) -> "RewriteResult":
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return cls.rewrite(fallback)

        if isinstance(raw, dict):
            try:
                payload = RewritePayload.model_validate(raw)
                return cls(type=payload.type, content=payload.content.strip())
            except ValidationError:
                pass

        return cls.rewrite(fallback)

    @classmethod
    def rewrite(cls, content: str) -> "RewriteResult":
        return cls(type="rewrite", content=content)

    @classmethod
    def clarify(cls, content: str) -> "RewriteResult":
        return cls(type="clarify", content=content)

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "content": self.content}


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ExecutorResponse:
    type: Literal["tool_call", "direct"]
    answer: str = ""
    raw_content: dict[str, Any] = field(default_factory=dict)
    tool_use_blocks: list[ToolUseBlock] = field(default_factory=list)
