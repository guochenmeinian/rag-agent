from .base import BaseTool
from websearch.src.fetch_web_content import WebContentFetcher
from websearch.src.retrieval import EmbeddingRetriever


class WebSearchTool(BaseTool):
    """Real-time web search using Serper + embedding retrieval."""

    name = "web_search"
    description = (
        "搜索实时网络信息，用于非蔚来专属查询（如竞品对比、实时资讯、通用知识）。"
        "不要用于蔚来车型参数查询，那些请用 rag_search。"
    )

    TOP_K = 5

    def run(self, query: str) -> str:
        fetcher = WebContentFetcher(query)
        contents, serper_response = fetcher.fetch()

        if not contents or not serper_response:
            return "未找到相关网络信息。"

        retriever = EmbeddingRetriever()
        docs = retriever.retrievel_embeddings(
            contents, serper_response["links"], query
        )

        if not docs:
            return "未找到相关内容。"

        parts = []
        for i, doc in enumerate(docs[: self.TOP_K]):
            url = doc.metadata.get("url", "")
            parts.append(f"[{i + 1}] {url}\n{doc.page_content}")

        return "\n\n".join(parts)

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词",
                    }
                },
                "required": ["query"],
            },
        }
