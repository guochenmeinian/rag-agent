import os
import time


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value in {"你的key", "your_key", "YOUR_KEY"}:
        raise RuntimeError(f"{name} is a placeholder value, please set a real key.")
    return value


def run_rag_e2e() -> None:
    # Required keys for end-to-end API test.
    require_env("LLAMA_CLOUD_API_KEY")
    require_env("OPENAI_API_KEY")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Delay heavy imports until required env vars are validated.
    from rag.generator import generate_answer
    from rag.pipeline import ingest, retrieve

    query = os.getenv("RAG_TEST_QUERY", "介绍一下技术规格")
    data_dir = os.getenv("RAG_DATA_DIR", "data")
    model = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
    limit = int(os.getenv("RAG_TOPK", "5"))

    col_name = f"rag_e2e_{int(time.time())}"
    print(f"[E2E] build index, collection={col_name}, data_dir={data_dir}")
    ctx = ingest(data_dir=data_dir, col_name=col_name)

    print(f"[E2E] retrieve, query={query!r}, topk={limit}")
    items = retrieve(query=query, ctx=ctx, limit=limit)
    if not items:
        raise RuntimeError("RAG retrieval returned no results.")

    non_empty = [x for x in items if x.get("text", "").strip()]
    if not non_empty:
        raise RuntimeError("RAG retrieval results are empty strings.")

    print("[E2E] generate answer via OpenAI")
    answer = generate_answer(query=query, items=items, model=model)
    if not answer.strip():
        raise RuntimeError("OpenAI returned empty answer.")

    print("\n=== Retrieved Snippets (Top 3) ===")
    for item in items[:3]:
        snippet = item["text"].replace("\n", " ")[:160]
        print(f"[{item['rank']}] {snippet}")

    print("\n=== Final Answer ===")
    print(answer)

if __name__ == "__main__":
    run_rag_e2e()
