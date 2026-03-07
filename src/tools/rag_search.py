from .base import BaseTool
from rag.pipeline import retrieve, format_citations, RAGContext


class RagSearchTool(BaseTool):
    """Search the NIO (蔚来) car knowledge base for vehicle specs and parameters."""

    name = "rag_search"
    description = (
        "从蔚来汽车知识库检索车型参数、规格、配置信息。"
        "仅用于蔚来车型相关查询，需指定具体车型。"
    )

    def __init__(self, contexts: dict[str, RAGContext]):
        # contexts: {"EC6": RAGContext, "ET5": RAGContext, ...}
        self.contexts = contexts

    def run(self, query: str, car_model: str) -> str:
        ctx = self.contexts.get(car_model)
        if ctx is None:
            available = list(self.contexts.keys())
            return f"未找到车型 '{car_model}' 的知识库。可用车型: {available}"

        results = retrieve(query, ctx)
        if not results:
            return f"未找到 {car_model} 关于「{query}」的相关信息。"

        return f"[{car_model}] {query} 检索结果：\n" + format_citations(results)

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
                        "description": "查询内容，如'电池容量'、'续航里程'、'充电功率'",
                    },
                    "car_model": {
                        "type": "string",
                        "description": "蔚来车型名称，如EC6、ET5、ES8、ET7、EL6",
                    },
                },
                "required": ["query", "car_model"],
            },
        }
