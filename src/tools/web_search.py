from .base import BaseTool


class WebSearchTool(BaseTool):
    """Placeholder — web search not yet implemented."""

    name = "web_search"
    description = (
        "搜索实时网络信息，用于非蔚来专属查询（如竞品对比、实时资讯、通用知识）。"
        "不要用于蔚来车型参数查询，那些请用 rag_search。"
    )

    def run(self, query: str) -> str:
        return "网络搜索功能暂未启用。"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询词",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
