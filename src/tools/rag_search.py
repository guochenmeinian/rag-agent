import threading
from typing import Literal

from pydantic import BaseModel, Field

from rag.pipeline import retrieve, format_citations, RAGContext
from .base import BaseTool
from .result import ToolResult

# Local Milvus uses a single-file DuckDB backend that cannot handle many
# concurrent connections.  This semaphore caps simultaneous Milvus queries
# across all threads (Planner + Executor combined) to avoid "Already borrowed"
# errors.  Switch to Milvus Docker to remove this limit.
_MILVUS_SEMAPHORE = threading.Semaphore(2)

MAX_CONTENT = 3000   # chars — prevents context overflow

# Valid NIO model codes; Pydantic rejects anything outside this set
CarModel = Literal["EC6", "EC7", "ES6", "ES8", "ET5", "ET5T", "ET7", "ET9"]


class RagSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="精炼的检索关键词，如'电池容量'、'CLTC续航里程'、'快充功率'、'轴距'",
        min_length=1,
        max_length=200,
    )
    car_model: CarModel = Field(
        ...,
        description="Nio车型代号，必须是以下之一: EC6 EC7 ES6 ES8 ET5 ET5T ET7 ET9",
    )


class RagSearchTool(BaseTool):
    """Retrieve NIO vehicle specs from the per-model Milvus collection."""

    name        = "rag_search"
    description = (
        "从Nio汽车知识库检索车型参数、规格、配置信息。"
        "仅用于Nio车型相关查询，需指定具体车型（car_model）。"
    )
    InputModel  = RagSearchInput
    timeout     = 30
    max_retries = 1
    cacheable   = True

    def __init__(self, contexts: dict[str, RAGContext]):
        self.contexts = contexts

    def _execute(self, inputs: RagSearchInput) -> ToolResult:
        with _MILVUS_SEMAPHORE:
            return self._query(inputs)

    def _query(self, inputs: RagSearchInput) -> ToolResult:
        ctx = self.contexts.get(inputs.car_model)
        if ctx is None:
            available = list(self.contexts.keys())
            return ToolResult.error(
                f"车型 '{inputs.car_model}' 的知识库未加载。已加载车型: {available}",
                "not_found",
            )

        results = retrieve(inputs.query, ctx)
        total_before = results[0].get("_total_before_filter", 0) if results else 0
        threshold    = results[0].get("_score_threshold", 0.0)   if results else 0.0

        if not results:
            return ToolResult(
                content=f"未找到 {inputs.car_model} 关于「{inputs.query}」的相关信息。",
                success=True,
                metadata={
                    "car_model": inputs.car_model,
                    "query": inputs.query,
                    "result_count": 0,
                    "total_before_filter": total_before,
                    "score_threshold": threshold,
                    "scores": [],
                },
            )

        content   = f"[{inputs.car_model}] {inputs.query} 检索结果：\n" + format_citations(results)
        truncated = len(content) > MAX_CONTENT

        return ToolResult(
            content=content[:MAX_CONTENT] + ("…[已截断]" if truncated else ""),
            success=True,
            truncated=truncated,
            metadata={
                "car_model":           inputs.car_model,
                "query":               inputs.query,
                "result_count":        len(results),
                "total_before_filter": total_before,
                "score_threshold":     threshold,
                "scores":              [r["score"] for r in results],
            },
        )
