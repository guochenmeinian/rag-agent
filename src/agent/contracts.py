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
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens


# ---------------------------------------------------------------------------
# Planner contracts
# ---------------------------------------------------------------------------

class _ToolCallSpecPayload(BaseModel):
    tool: Literal["rag_search", "grep_search", "web_search"]
    query: str = Field(..., min_length=1)
    car_model: str | None = None


class _PlanResultPayload(BaseModel):
    type: Literal["simple", "decomposed"]
    calls: list[_ToolCallSpecPayload] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolCallSpec:
    tool: str
    query: str
    car_model: str | None = None


@dataclass(frozen=True)
class PlanResult:
    type: Literal["simple", "decomposed"]
    calls: tuple[ToolCallSpec, ...]  # frozen-safe container

    @classmethod
    def parse(cls, raw: str | dict[str, Any] | None) -> "PlanResult":
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return cls.simple()
        if isinstance(raw, dict):
            try:
                payload = _PlanResultPayload.model_validate(raw)
                specs = tuple(
                    ToolCallSpec(tool=c.tool, query=c.query, car_model=c.car_model)
                    for c in payload.calls
                )
                # Treat decomposed with no calls as simple
                if payload.type == "decomposed" and not specs:
                    return cls.simple()
                return cls(type=payload.type, calls=specs)
            except ValidationError:
                pass
        return cls.simple()

    @classmethod
    def simple(cls) -> "PlanResult":
        return cls(type="simple", calls=())

    @classmethod
    def decomposed(cls, calls: list[ToolCallSpec]) -> "PlanResult":
        return cls(type="decomposed", calls=tuple(calls))
