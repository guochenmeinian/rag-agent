"""Grep 风格关键词检索 — 精确匹配参数、型号、数字等。"""
from typing import Literal

from pydantic import BaseModel, Field

import config
from storage.grep_index import GrepIndex
from .base import BaseTool
from .result import ToolResult

MAX_CONTENT = 3000
CarModel = Literal["EC6", "EC7", "ES6", "ES8", "ET5", "ET5T", "ET7", "ET9"]


class GrepSearchInput(BaseModel):
    keywords: str = Field(
        ...,
        description="精确关键词，空格分隔，如'轴距 毫米'、'100kWh CLTC'、'快充功率'",
        min_length=1,
        max_length=200,
    )
    car_model: CarModel = Field(
        ...,
        description="蔚来车型代号: EC6 EC7 ES6 ES8 ET5 ET5T ET7 ET9",
    )


class GrepSearchTool(BaseTool):
    """
    关键词全文检索（grep 风格），适合精确参数查询。
    当问题包含具体型号、数字、参数名时，可与 rag_search 互补或并行使用。
    """

    name = "grep_search"
    description = (
        "按关键词在车型手册中精确检索，适合具体参数（轴距、续航、电池容量等）。"
        "需指定 car_model。与 rag_search 互补：精确参数用 grep，概念/对比用 rag。"
    )
    InputModel = GrepSearchInput
    timeout = 10
    max_retries = 1
    cacheable = True

    def __init__(self, db_path: str | None = None):
        self.index = GrepIndex(db_path or config.GREP_INDEX_PATH)

    def _execute(self, inputs: GrepSearchInput) -> ToolResult:
        col_name = f"nio_{inputs.car_model.lower()}"
        results = self.index.search(col_name, inputs.keywords, limit=5)

        if not results:
            return ToolResult(
                content=f"grep 未找到 {inputs.car_model} 关于「{inputs.keywords}」的匹配片段。",
                success=True,
                metadata={
                    "car_model": inputs.car_model,
                    "keywords": inputs.keywords,
                    "result_count": 0,
                },
            )

        lines = [
            f"[{i+1}] {r['text']}"
            for i, r in enumerate(results)
        ]
        content = f"[{inputs.car_model}] grep「{inputs.keywords}」结果：\n" + "\n\n".join(lines)
        truncated = len(content) > MAX_CONTENT

        return ToolResult(
            content=content[:MAX_CONTENT] + ("…[已截断]" if truncated else ""),
            success=True,
            truncated=truncated,
            metadata={
                "car_model": inputs.car_model,
                "keywords": inputs.keywords,
                "result_count": len(results),
            },
        )
