import re

import requests
from pydantic import BaseModel, Field

import config
from .base import BaseTool
from .result import ToolResult

_SERPER_URL = "https://google.serper.dev/search"
_TOP_K      = 5
_MAX_CONTENT = 2000


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="搜索查询词，自然语言，中英文均可",
        min_length=1,
        max_length=300,
    )


class WebSearchTool(BaseTool):
    """Lightweight Serper-based web search (snippets only, no crawling)."""

    name        = "web_search"
    description = (
        "搜索实时网络信息，用于非Nio专属查询（竞品对比、实时资讯、通用知识）。"
        "不要用于Nio车型参数查询，那些请用 rag_search。"
    )
    InputModel  = WebSearchInput
    timeout     = 15
    max_retries = 2
    cacheable   = False

    def _execute(self, inputs: WebSearchInput) -> ToolResult:
        if not config.SERPER_API_KEY:
            return ToolResult.error(
                "网络搜索需要 SERPER_API_KEY，请在 .env 中配置后重启。当前仅支持知识库查询。",
                "not_configured",
            )
        try:
            return self._search(inputs.query)
        except requests.Timeout:
            return ToolResult.error("网络搜索请求超时", "timeout")
        except Exception as exc:
            return ToolResult.error(str(exc), "execution_error")

    def _search(self, query: str) -> ToolResult:
        payload: dict = {"q": query, "num": _TOP_K}
        if _has_chinese(query):
            payload.update({"gl": "cn", "hl": "zh-cn"})

        resp = requests.post(
            _SERPER_URL,
            headers={"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        items: list[str] = []

        kg = data.get("knowledgeGraph", {})
        if kg.get("description"):
            items.append(f"[概要] {kg['title']}: {kg['description']}")

        for r in data.get("organic", [])[:_TOP_K]:
            snippet = r.get("snippet", "")
            if snippet:
                items.append(f"• {r.get('title', '')}\n  {snippet}")

        if not items:
            return ToolResult(
                content=f"网络搜索「{query}」未找到相关结果。",
                success=True,
                metadata={"query": query, "result_count": 0},
            )

        content   = f"网络搜索「{query}」结果：\n\n" + "\n\n".join(items)
        truncated = len(content) > _MAX_CONTENT

        return ToolResult(
            content=content[:_MAX_CONTENT] + ("…[已截断]" if truncated else ""),
            success=True,
            truncated=truncated,
            metadata={"query": query, "result_count": len(items)},
        )
