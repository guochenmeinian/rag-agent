"""Entry point for the NIO car assistant agent (REPL mode).

Usage:
    python src/main.py [--session SESSION_ID]

Environment variables required:
    OPENAI_API_KEY    - OpenAI API key (GPT-4o executor)
    DASHSCOPE_API_KEY - Alibaba DashScope key (Qwen rewriter/reflector/memory)

Optional:
    SERPER_API_KEY    - Serper key for web_search tool
    OPENAI_MODEL      - GPT model ID (default: gpt-4o)
    QWEN_MODEL        - Qwen model ID (default: qwen3.5-instruct)
    NIO_CAR_MODELS    - Comma-separated car models (default: EC6,EC7,ES6,ES8,ET5,ET5T,ET7,ET9)
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: loads .env
from agent.workflow import AgentWorkflow
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from rag.pipeline import ingest, RAGContext


def load_rag_contexts() -> dict[str, RAGContext]:
    contexts = {}
    
    for model in config.NIO_CAR_MODELS:
        pdf_path = os.path.join(config.DATA_ROOT, f"{model}.pdf")
        
        if not os.path.isfile(pdf_path):
            print(f"[warn] No PDF file for {model} at {pdf_path}, skipping.")
            continue
        
        print(f"[info] Ingesting {model}...")
        col_name = f"nio_{model.lower()}"
        
        ctx = ingest(
            data_dir=config.DATA_ROOT,
            uri=config.MILVUS_URI,
            col_name=col_name,
            file_filter=f"{model}.pdf"
        )
        contexts[model] = ctx
        print(f"[info] {model} ready.")
    
    return contexts


def build_registry(rag_contexts: dict[str, RAGContext]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    if rag_contexts:
        registry.register(RagSearchTool(contexts=rag_contexts))
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None, help="Session ID for memory persistence")
    args = parser.parse_args()

    rag_contexts = load_rag_contexts()
    registry = build_registry(rag_contexts)
    user_profile = os.getenv("USER_PROFILE", "")

    workflow = AgentWorkflow(
        registry=registry,
        user_profile=user_profile,
        session_id=args.session,
    )

    if args.session:
        print(f"[info] Session: {args.session}")

    print("蔚来汽车助手已启动，输入 'quit' 退出\n")
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit", "退出"):
            break

        answer = workflow.run(user_input)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()