"""ToolRegistry — parallel execution with per-tool timeout and retry."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from .base import BaseTool
from .result import ToolResult

# ── Structured tool call format ────────────────────────────────────────────────
# Every call dict flowing through the system has the shape:
#   {
#     "id":    str,        # OpenAI tool_call_id
#     "name":  str,        # tool name
#     "input": dict,       # raw kwargs from LLM (validated inside tool.run())
#   }
#
# Every result dict has the shape:
#   {
#     "id":     str,        # echo of call["id"]
#     "name":   str,        # echo of call["name"]
#     "query":  str,        # human-readable label (call["input"]["query"])
#     "result": ToolResult, # structured output
#   }
# ──────────────────────────────────────────────────────────────────────────────


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    # ── Registration ──────────────────────────────────────────

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def schemas(self) -> list[dict]:
        return [t.schema for t in self._tools.values()]

    # ── Parallel execution ────────────────────────────────────

    def run_parallel(self, calls: list[dict]) -> list[dict]:
        """Execute all tool calls in parallel, preserving original order.

        Each call: {"id": str, "name": str, "input": dict}
        Returns:   [{"id", "name", "query", "result": ToolResult}, ...]
        """
        if not calls:
            return []

        results: list[dict | None] = [None] * len(calls)

        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            future_to_idx = {
                pool.submit(self._run_with_resilience, call): i
                for i, call in enumerate(calls)
            }
            for future in as_completed(future_to_idx):
                idx  = future_to_idx[future]
                call = calls[idx]
                try:
                    tool_result = future.result()
                except Exception as exc:
                    tool_result = ToolResult.error(str(exc))

                results[idx] = {
                    "id":    call["id"],
                    "name":  call["name"],
                    "query": call["input"].get("query", ""),
                    "result": tool_result,
                }

        return results  # type: ignore[return-value]

    # ── Internal: resilience wrapper ──────────────────────────

    def _run_with_resilience(self, call: dict) -> ToolResult:
        """Run one tool call with timeout and exponential-backoff retry."""
        tool        = self.get(call["name"])
        timeout_s   = getattr(tool, "timeout",     30)
        max_retries = getattr(tool, "max_retries",  1)

        last_result: ToolResult | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                backoff = min(2 ** (attempt - 1), 8)   # 1s, 2s, 4s, …
                time.sleep(backoff)

            try:
                last_result = self._run_with_timeout(tool, call["input"], timeout_s)
            except FuturesTimeout:
                last_result = ToolResult.error(
                    f"执行超时（{timeout_s}s），请告知用户该信息暂时无法获取",
                    "timeout",
                )
                # Don't retry timeouts — they'd just block again
                return last_result
            except Exception as exc:
                last_result = ToolResult.error(str(exc), "execution_error")

            # Stop early if successful or a validation error (retrying won't help)
            if last_result.success or last_result.error_type == "validation_error":
                return last_result

        return last_result or ToolResult.error("Unknown error after retries")

    @staticmethod
    def _run_with_timeout(tool: BaseTool, kwargs: dict, timeout_s: int) -> ToolResult:
        """Execute tool.run() in a separate thread with a hard timeout."""
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(tool.run, **kwargs)
            return future.result(timeout=timeout_s)   # raises FuturesTimeout if exceeded
