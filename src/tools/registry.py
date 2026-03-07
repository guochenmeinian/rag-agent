from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return self._tools[name]

    @property
    def schemas(self) -> list[dict]:
        return [t.schema for t in self._tools.values()]

    def run_parallel(self, calls: list[dict]) -> list[dict]:
        """Run multiple tool calls in parallel.

        Args:
            calls: list of {id, name, input} dicts (from Claude's tool_use blocks)

        Returns:
            list of {id, name, query, result} dicts in original order
        """
        results = [None] * len(calls)

        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            future_to_idx = {
                executor.submit(self._run_one, call): i
                for i, call in enumerate(calls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                call = calls[idx]
                try:
                    result = future.result()
                except Exception as e:
                    result = f"Tool error: {e}"
                results[idx] = {
                    "id": call["id"],
                    "name": call["name"],
                    "query": call["input"].get("query", ""),
                    "result": result,
                }

        return results

    def _run_one(self, call: dict) -> str:
        tool = self.get(call["name"])
        return tool.run(**call["input"])
