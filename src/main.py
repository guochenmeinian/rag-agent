"""Entry point for the NIO car assistant agent.

Usage:
    python src/main.py

Environment variables required:
    ANTHROPIC_API_KEY   - Claude API key
    DASHSCOPE_API_KEY   - Alibaba DashScope API key (for Qwen)
    SERPER_API_KEY      - Serper search API key (used by websearch module)

Optional:
    QWEN_MODEL          - Qwen model ID (default: qwen3.5-instruct)
    CLAUDE_MODEL        - Claude model ID (default: claude-sonnet-4-6)

RAG setup (per car model):
    Each NIO car model needs a pre-built Milvus collection.
    Call `rag.pipeline.ingest(data_dir, uri, col_name)` first.
"""

import sys
import os

# Ensure src/ is in path for intra-package imports
sys.path.insert(0, os.path.dirname(__file__))

from agent.workflow import AgentWorkflow
from agent.responder import format_answer
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from rag.pipeline import ingest, RAGContext


def build_registry(rag_contexts: dict[str, RAGContext]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(RagSearchTool(contexts=rag_contexts))
    return registry


def load_rag_contexts(
    car_models: list[str],
    data_root: str = "data",
    db_uri: str = "./milvus.db",
) -> dict[str, RAGContext]:
    """Ingest data for each car model into its own Milvus collection."""
    contexts = {}
    for model in car_models:
        data_dir = os.path.join(data_root, model)
        if not os.path.exists(data_dir):
            print(f"[warn] No data directory for {model}: {data_dir}, skipping.")
            continue
        print(f"[info] Ingesting {model} ...")
        col_name = f"nio_{model.lower()}"
        ctx = ingest(data_dir=data_dir, uri=db_uri, col_name=col_name)
        contexts[model] = ctx
        print(f"[info] {model} ready.")
    return contexts


def main():
    # --- Setup ---
    car_models = ["EC6", "ET5", "ES8", "ET7", "EL6"]
    rag_contexts = load_rag_contexts(car_models)

    registry = build_registry(rag_contexts)

    user_profile = os.getenv("USER_PROFILE", "")
    workflow = AgentWorkflow(registry=registry, user_profile=user_profile)

    # --- REPL ---
    print("蔚来汽车助手已启动，输入 'quit' 退出\n")
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit", "退出"):
            break

        answer = workflow.run(user_input)
        formatted = format_answer(
            answer=answer,
            refined_query=workflow.memory.recent_messages[-1]["content"]
            if workflow.memory.recent_messages
            else user_input,
            tool_results=None,  # responder gets answer post-hoc; pass state if needed
        )
        print(f"\nAssistant: {formatted}\n")


if __name__ == "__main__":
    main()
