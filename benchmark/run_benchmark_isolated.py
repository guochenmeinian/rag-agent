"""Isolated (unit-test style) benchmark — each module tested independently.

Unlike run_benchmark.py (end-to-end), upstream ground-truth data is injected
so that only the target module is exercised.

    --layer rewriter   → QueryRewriter only; no executor, no tools (cheap: Qwen only)
    --layer router     → Executor called once; tool SELECTION only, tools NOT executed
    --layer retrieval  → Tools called directly with GT query_intent; no rewriter/executor

Usage
─────
python benchmark/run_benchmark_isolated.py --layer rewriter
python benchmark/run_benchmark_isolated.py --layer router
python benchmark/run_benchmark_isolated.py --layer retrieval
python benchmark/run_benchmark_isolated.py --layer rewriter --delay 2 --out results/iso_rw.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import config as src_config
import benchmark.config as bm_config
from benchmark.eval.rewriter_eval import eval_rewriter_case, aggregate_rewriter
from benchmark.eval.router_eval import eval_router_case, aggregate_router
from benchmark.eval.retrieval_eval import eval_retrieval_case, aggregate_retrieval


# ─────────────────────────────────────────────────────────────
# Helpers shared with run_benchmark
# ─────────────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def _build_memory(case: dict):
    """Construct a ConversationMemory from a case's context field."""
    from agent.memory import ConversationMemory
    mem = ConversationMemory()
    ctx = case.get("context") or {}
    if ctx.get("user_profile"):
        mem.user_profile = ctx["user_profile"]
        mem.global_user_info.raw = ctx["user_profile"]
    for fact in ctx.get("memory_facts", []):
        mem.facts.append(fact)
    for msg in ctx.get("conversation_history", []):
        mem.add_message(msg["role"], msg["content"])
    return mem


def load_rag_contexts():
    from rag.pipeline import ingest
    contexts = {}
    for model in src_config.NIO_CAR_MODELS:
        subdir = os.path.join(src_config.DATA_ROOT, model)
        flat_pdf = os.path.join(src_config.DATA_ROOT, f"{model}.pdf")
        col_name = f"nio_{model.lower()}"
        try:
            if os.path.isdir(subdir):
                ctx = ingest(data_dir=subdir, uri=src_config.MILVUS_URI, col_name=col_name)
            elif os.path.isfile(flat_pdf):
                ctx = ingest(data_dir=src_config.DATA_ROOT, uri=src_config.MILVUS_URI,
                             col_name=col_name, file_filter=f"{model}.pdf")
            else:
                continue
            contexts[model] = ctx
        except Exception as exc:
            print(f"[warn] RAG context for {model} failed: {exc}")
    return contexts


# ─────────────────────────────────────────────────────────────
# REWRITER — isolated
# ─────────────────────────────────────────────────────────────

def run_rewriter_isolated(case: dict) -> dict:
    """Call QueryRewriter directly. No executor, no tools."""
    if "rewriter" not in case.get("layer_targets", []) or not case.get("rewriter_gt"):
        return {"id": case["id"], "skip": True, "reason": "no rewriter_gt"}

    from agent.rewriter import QueryRewriter
    rewriter = QueryRewriter(**(src_config.get_qwen_cfg()))
    mem = _build_memory(case)
    context_prompt = mem.format_for_prompt()

    t0 = time.monotonic()
    try:
        result = rewriter.rewrite(case["input"], context_prompt)
    except Exception as exc:
        return {"id": case["id"], "error": str(exc), "input": case["input"]}
    latency = round(time.monotonic() - t0, 2)

    rewrite_result = {"type": result.type, "content": result.content}
    r = eval_rewriter_case(case, rewrite_result)
    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "input": case["input"],
        "latency_s": latency,
        "metrics": {f"rewriter/{k}": v for k, v in r["metrics"].items()},
        "detail": {"rewriter": r["detail"]},
        "rewrite_result": rewrite_result,
    }


# ─────────────────────────────────────────────────────────────
# ROUTER — isolated (tool selection only, tools NOT executed)
# ─────────────────────────────────────────────────────────────

def run_router_isolated(case: dict) -> dict:
    """Run executor once to get tool selection. Tools are NOT called."""
    if "router" not in case.get("layer_targets", []) or not case.get("router_gt"):
        return {"id": case["id"], "skip": True, "reason": "no router_gt"}

    from agent.executor import AgentExecutor
    from tools.registry import ToolRegistry
    from tools.rag_search import RagSearchTool
    from tools.grep_search import GrepSearchTool
    from tools.web_search import WebSearchTool

    # Build registry with all tools so schemas are available, but we won't run them
    registry = ToolRegistry()
    registry.register(RagSearchTool(contexts={}))   # empty contexts — schemas only
    registry.register(GrepSearchTool())
    registry.register(WebSearchTool())

    executor = AgentExecutor(tool_schemas=registry.schemas, **(src_config.get_executor_cfg()))
    mem = _build_memory(case)
    context_prompt = mem.format_for_prompt()

    # Use the raw input (same as what a real router would see after rewriting)
    # For true isolation of the router, we use raw input since router cases
    # are designed with clear, unambiguous queries.
    messages = [{"role": "user", "content": case["input"]}]

    t0 = time.monotonic()
    try:
        response = executor.run(messages, extra_system=context_prompt)
    except Exception as exc:
        return {"id": case["id"], "error": str(exc), "input": case["input"]}
    latency = round(time.monotonic() - t0, 2)

    if response.type == "tool_call":
        actual_calls = [{"name": b.name, **b.input} for b in response.tool_use_blocks]
        batches = [actual_calls]   # single batch (one LLM response)
    else:
        actual_calls = []
        batches = []

    r = eval_router_case(case, actual_calls, batches)
    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "input": case["input"],
        "latency_s": latency,
        "metrics": {f"router/{k}": v for k, v in r["metrics"].items()},
        "detail": {"router": r["detail"]},
        "actual_calls": actual_calls,
    }


# ─────────────────────────────────────────────────────────────
# RETRIEVAL — isolated (tools called directly with GT query)
# ─────────────────────────────────────────────────────────────

def run_retrieval_isolated(case: dict, rag_contexts: dict) -> dict:
    """Call RAG/grep tools directly using GT query_intent. No LLM calls at all."""
    if "retrieval" not in case.get("layer_targets", []) or not case.get("retrieval_gt"):
        return {"id": case["id"], "skip": True, "reason": "no retrieval_gt"}

    from tools.rag_search import RagSearchTool
    from tools.grep_search import GrepSearchTool

    ret_gt = case["retrieval_gt"]
    query_intent = ret_gt.get("query_intent", case["input"])
    expected_facts = ret_gt.get("expected_facts", [])

    # Identify target car model(s) from expected_facts
    car_models = [f for f in expected_facts if f in src_config.NIO_CAR_MODELS]
    if not car_models:
        # Try to parse from query_intent
        car_models = [m for m in src_config.NIO_CAR_MODELS if m in query_intent]
    if not car_models:
        car_models = list(rag_contexts.keys())[:1]

    rag_tool = RagSearchTool(contexts=rag_contexts)
    grep_tool = GrepSearchTool()

    chunks_out: list[dict] = []
    t0 = time.monotonic()
    try:
        for car_model in car_models:
            # RAG search with GT intent
            rag_result = rag_tool.run(query=query_intent, car_model=car_model)
            if rag_result and rag_result.success and rag_result.content:
                chunks_out.append({
                    "id": f"rag_search:{car_model}",
                    "content": rag_result.content,
                })
            # Grep search with key terms from expected_facts (non-model facts)
            grep_keywords = " ".join(f for f in expected_facts if f not in src_config.NIO_CAR_MODELS)
            if grep_keywords:
                grep_result = grep_tool.run(keywords=grep_keywords, car_model=car_model)
                if grep_result and grep_result.success and grep_result.content:
                    chunks_out.append({
                        "id": f"grep_search:{car_model}",
                        "content": grep_result.content,
                    })
    except Exception as exc:
        return {"id": case["id"], "error": str(exc), "input": case["input"]}
    latency = round(time.monotonic() - t0, 2)

    r = eval_retrieval_case(case, chunks_out)
    if r.get("skip"):
        return {"id": case["id"], "skip": True, "reason": "eval skipped"}

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "input": case["input"],
        "latency_s": latency,
        "metrics": {f"retrieval/{k}": v for k, v in r["metrics"].items()},
        "detail": {"retrieval": {**r.get("detail", {}), "mode": r.get("mode")}},
    }


# ─────────────────────────────────────────────────────────────
# Aggregation & Reporting
# ─────────────────────────────────────────────────────────────

def _strip_prefix(metrics: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in metrics.items() if k.startswith(prefix)}


def aggregate(results: list[dict], layer: str) -> dict:
    valid = [r for r in results if not r.get("error") and not r.get("skip") and not r.get("dry_run")]
    prefix = f"{layer}/"

    flat = [
        {"id": r["id"], "category": r["category"],
         "metrics": _strip_prefix(r["metrics"], prefix),
         "detail":  r.get("detail", {}).get(layer, {})}
        for r in valid if any(k.startswith(prefix) for k in r.get("metrics", {}))
    ]

    if layer == "rewriter":
        summary = aggregate_rewriter(flat)
    elif layer == "router":
        summary = aggregate_router(flat)
    elif layer == "retrieval":
        ret_detail = lambda r: r.get("detail", {}).get("retrieval", {})
        full = [{"id": r["id"], "input": r["input"], "skip": False,
                 "metrics":      _strip_prefix(r["metrics"], prefix),
                 "detail":       ret_detail(r),
                 "mode":         ret_detail(r).get("mode"),
                 "expect_no_hit": ret_detail(r).get("expect_no_hit", False)}
                for r in valid if any(k.startswith(prefix) for k in r.get("metrics", {}))]
        summary = aggregate_retrieval(full)
    else:
        summary = {}

    latencies = [r["latency_s"] for r in valid if "latency_s" in r]
    return {
        "layer": layer,
        "mode":  "isolated",
        "n_total": len(results),
        "n_run":   len(valid),
        "n_skip":  sum(1 for r in results if r.get("skip")),
        "n_error": sum(1 for r in results if r.get("error")),
        layer: summary,
        "latency": {
            "p50": sorted(latencies)[int(len(latencies) * 0.5)],
            "p95": sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)],
        } if latencies else {},
    }


def print_summary(summary: dict, layer: str):
    print("\n" + "=" * 65)
    print(f"  ISOLATED BENCHMARK — {layer.upper()}")
    print("=" * 65)
    print(f"  Cases: {summary['n_run']} run / {summary.get('n_skip', 0)} skip / {summary.get('n_error', 0)} errors\n")

    METRICS = {
        "rewriter":  ["standalone", "entity_extraction_accuracy", "clarify_detection",
                      "coref_resolution_rate", "ellipsis_fill_rate"],
        "router":    ["tool_classification_accuracy", "parameter", "multi_query", "avg_duplicate_calls"],
        "retrieval": ["hit@1", "hit@3", "hit@5", "mrr", "relevance@5", "facts_coverage_avg", "no_hit_ok"],
    }

    data = summary.get(layer, {})
    for m in METRICS.get(layer, []):
        if m not in data:
            continue
        val = data[m]
        val_str = "N/A" if val is None else f"{val:.3f}"
        print(f"  {m:<40} {val_str}")

    if summary.get("latency"):
        lat = summary["latency"]
        print(f"\n  LATENCY  p50={lat['p50']}s  p95={lat['p95']}s")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Isolated module benchmark")
    p.add_argument("--layer",   required=True, choices=["rewriter", "router", "retrieval"])
    p.add_argument("--dataset", default=str(bm_config.CASES_FILE))
    p.add_argument("--out",     default="")
    p.add_argument("--delay",   type=float, default=2.0,
                   help="Seconds between cases (default: 2.0). Router uses gpt-4o; set higher if 429.")
    p.add_argument("--ids",     default="", help="Comma-separated case IDs to filter")
    return p.parse_args()


def main():
    args = parse_args()
    cases = load_dataset(args.dataset)

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    # Filter to cases that target this layer
    cases = [c for c in cases if args.layer in c.get("layer_targets", [])]

    if not cases:
        print(f"No cases targeting layer '{args.layer}'.")
        return

    print(f"\nIsolated [{args.layer.upper()}] — {len(cases)} cases\n")

    rag_contexts = {}
    if args.layer == "retrieval":
        print("Loading RAG contexts...")
        rag_contexts = load_rag_contexts()
        print(f"Loaded: {list(rag_contexts.keys())}\n")

    results = []
    for i, case in enumerate(cases):
        if i > 0 and args.delay > 0:
            time.sleep(args.delay)

        print(f"  → {case['id']} : {case['input'][:60]}")
        if args.layer == "rewriter":
            r = run_rewriter_isolated(case)
        elif args.layer == "router":
            r = run_router_isolated(case)
        elif args.layer == "retrieval":
            r = run_retrieval_isolated(case, rag_contexts)
        else:
            r = {"id": case["id"], "skip": True, "reason": "unsupported"}

        results.append(r)

        if r.get("skip"):
            print(f"  [SKIP] {r['id']} — {r.get('reason', '')}")
        elif r.get("error"):
            print(f"  [ERR ] {r['id']} — {r['error'][:70]}")
        else:
            metrics_str = "  ".join(
                f"{k.split('/')[1][:8]}={v if v is None else round(v, 3)}"
                for k, v in list(r.get("metrics", {}).items())[:4]
            )
            print(f"  [OK  ] {r['id']:<35} {metrics_str}  {r.get('latency_s', '?')}s")

    summary = aggregate(results, args.layer)
    print_summary(summary, args.layer)

    out_path = args.out or str(bm_config.RESULTS_DIR / f"isolated_{args.layer}_{int(time.time())}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
