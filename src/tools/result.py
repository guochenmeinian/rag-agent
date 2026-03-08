"""ToolResult — structured output contract for all tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolResult:
    """Structured return type for every tool execution.

    content     — text handed to the LLM (role=tool message)
    success     — False signals an error the LLM should reason about
    error_type  — machine-readable error class for retry/fallback logic
    metadata    — arbitrary extra data (source, car_model, result_count …)
    latency_ms  — filled in by BaseTool.run() after execution
    truncated   — True if content was clipped to MAX_CONTENT
    """
    content: str
    success: bool = True
    error_type: Optional[str] = None   # validation_error | timeout | not_found | not_configured | execution_error
    metadata: dict = field(default_factory=dict)
    latency_ms: int = 0
    truncated: bool = False

    # ── Convenience constructors ──────────────────────────────

    @classmethod
    def ok(cls, content: str, truncated: bool = False, **metadata) -> ToolResult:
        return cls(content=content, success=True, truncated=truncated, metadata=metadata)

    @classmethod
    def error(cls, message: str, error_type: str = "execution_error") -> ToolResult:
        return cls(content=message, success=False, error_type=error_type)

    # ── Utilities ─────────────────────────────────────────────

    def to_llm_content(self) -> str:
        """Content string that goes into the OpenAI tool message."""
        if not self.success:
            # Prefix errors so the LLM can reason about them
            prefix = {
                "validation_error":  "[工具参数错误]",
                "timeout":           "[工具超时]",
                "not_found":         "[未找到数据]",
                "not_configured":    "[工具未配置]",
                "execution_error":   "[工具执行失败]",
            }.get(self.error_type or "", "[工具错误]")
            return f"{prefix} {self.content}"
        return self.content

    def __str__(self) -> str:
        return self.content
