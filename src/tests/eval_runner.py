"""Eval runner for the NIO assistant.

Usage:
    cd src && python -m tests.eval_runner [--ids q001,q004] [--category multi_car_compare]

Metrics reported:
    tool_precision   fraction of expected tools that were actually called
    tool_recall      fraction of called tools that were expected
    parallel_ok      for cases with expected_parallel=True, did model issue multiple calls at once?
    keyword_hit      fraction of expected_answer_keywords found in final answer
    hallucination    any must_not_contain items found in answer (lower is better)
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config  # noqa: loads .env
from agent.workflow import AgentWorkflow
from agent.memory import ConversationMemory
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from rag.pipeline import ingest, RAGContext


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


# ──────────────────────────────────────────────
# Workflow factory
# ──────────────────────────────────────────────

def build_workflow(rag_contexts: dict[str, RAGContext]) -> AgentWorkflow:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    if rag_contexts:
        registry.register(RagSearchTool(contexts=rag_contexts))
    return AgentWorkflow(registry=registry)


def load_rag_contexts() -> dict[str, RAGContext]:
    contexts = {}
    for model in config.NIO_CAR_MODELS:
        subdir = os.path.join(config.DATA_ROOT, model)
        flat_pdf = os.path.join(config.DATA_ROOT, f"{model}.pdf")
        col_name = f"nio_{model.lower()}"
        try:
            if os.path.isdir(subdir):
                ctx = ingest(data_dir=subdir, uri=config.MILVUS_URI, col_name=col_name)
            elif os.path.isfile(flat_pdf):
                import shutil, tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    shutil.copy(flat_pdf, os.path.join(tmp, f"{model}.pdf"))
                    ctx = ingest(data_dir=tmp, uri=config.MILVUS_URI, col_name=col_name)
            else:
                continue
            contexts[model] = ctx
        except Exception as e:
            print(f"[warn] {model}: {e}")
    return contexts


# ──────────────────────────────────────────────
# Per-case evaluation
# ──────────────────────────────────────────────

def inject_context(workflow: AgentWorkflow, context: dict | None):
    """Inject pre-built memory context to simulate mid-conversation state."""
    if not context:
        return
    if "summary" in context:
        workflow.memory.context_summary = context["summary"]
    for msg in context.get("recent_messages", []):
        workflow.memory.recent_messages.append(msg)


def eval_case(case: dict, workflow: AgentWorkflow) -> dict:
    # Reset memory for each case
    workflow.memory = ConversationMemory(
        system_prompt=workflow.memory.system_prompt,
        user_profile=workflow.memory.user_profile,
    )
    inject_context(workflow, case.get("context"))

    # Collect events
    tool_calls_issued: list[dict] = []   # {name, car_model?, query?}
    tool_batches: list[list] = []        # each inner list = one parallel batch
    refined_query = case["input"]
    answer = ""

    current_batch: list[dict] = []
    for event in workflow.run_stream(case["input"]):
        if event["type"] == "refined":
            refined_query = event["query"]
        elif event["type"] == "tool_calling":
            current_batch = event["calls"]
            tool_calls_issued.extend(current_batch)
            tool_batches.append(list(current_batch))
        elif event["type"] == "done":
            answer = event["answer"]

    # ── Metrics ──
    result = {
        "id": case["id"],
        "category": case["category"],
        "input": case["input"],
        "refined_query": refined_query,
        "answer_preview": answer[:200],
        "tools_called": [c.get("name") for c in tool_calls_issued],
        "tool_details": tool_calls_issued,
    }

    # Tool precision / recall
    expected_tools = case.get("expected_tools", [])
    expected_names = [t["name"] for t in expected_tools]
    called_names = [c.get("name") for c in tool_calls_issued]

    if expected_names:
        hits = sum(1 for n in expected_names if n in called_names)
        result["tool_recall"] = hits / len(expected_names)
    else:
        result["tool_recall"] = 1.0 if not called_names else 0.0

    if called_names:
        hits = sum(1 for n in called_names if n in expected_names)
        result["tool_precision"] = hits / len(called_names)
    else:
        result["tool_precision"] = 1.0 if not expected_names else 0.0

    # Parallel check
    if case.get("expected_parallel"):
        max_batch = max((len(b) for b in tool_batches), default=0)
        result["parallel_ok"] = max_batch >= 2
    else:
        result["parallel_ok"] = None

    # car_model accuracy
    expected_models = [t.get("car_model") for t in expected_tools if t.get("car_model")]
    called_models = [c.get("car_model") for c in tool_calls_issued if c.get("car_model")]
    if expected_models:
        model_hits = sum(1 for m in expected_models if m in called_models)
        result["car_model_recall"] = model_hits / len(expected_models)
    else:
        result["car_model_recall"] = None

    # Rewrite quality (keyword check)
    rewrite_keywords = case.get("expected_rewrite_contains", [])
    if rewrite_keywords:
        rw_hits = sum(1 for kw in rewrite_keywords if kw in refined_query)
        result["rewrite_keyword_hit"] = rw_hits / len(rewrite_keywords)
    else:
        result["rewrite_keyword_hit"] = None

    # Answer keyword coverage
    kws = case.get("expected_answer_keywords", [])
    if kws:
        kw_hits = sum(1 for kw in kws if kw.lower() in answer.lower())
        result["keyword_hit"] = kw_hits / len(kws)
    else:
        result["keyword_hit"] = None

    # Hallucination check
    bad = [s for s in case.get("must_not_contain", []) if s.lower() in answer.lower()]
    result["hallucination_hits"] = bad
    result["pass"] = (
        result["tool_recall"] >= 0.8
        and result["tool_precision"] >= 0.8
        and not bad
        and (result["parallel_ok"] is not False)
    )

    return result


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def print_report(results: list[dict]):
    total = len(results)
    passed = sum(1 for r in results if r["pass"])

    print("\n" + "=" * 70)
    print(f"EVAL RESULTS  {passed}/{total} passed")
    print("=" * 70)

    for r in results:
        status = "✅ PASS" if r["pass"] else "❌ FAIL"
        print(f"\n[{r['id']}] {status}  {r['category']}")
        print(f"  Input  : {r['input']}")
        print(f"  Rewrite: {r['refined_query']}")
        print(f"  Tools  : {r['tools_called']}")
        print(f"  tool_recall={r['tool_recall']:.2f}  tool_precision={r['tool_precision']:.2f}", end="")
        if r["parallel_ok"] is not None:
            print(f"  parallel={'✅' if r['parallel_ok'] else '❌'}", end="")
        if r["car_model_recall"] is not None:
            print(f"  car_model_recall={r['car_model_recall']:.2f}", end="")
        if r["keyword_hit"] is not None:
            print(f"  keyword_hit={r['keyword_hit']:.2f}", end="")
        if r["rewrite_keyword_hit"] is not None:
            print(f"  rewrite_hit={r['rewrite_keyword_hit']:.2f}", end="")
        print()
        if r["hallucination_hits"]:
            print(f"  ⚠ hallucination: {r['hallucination_hits']}")
        print(f"  Answer : {r['answer_preview'][:120]}...")

    # Aggregate
    print("\n" + "-" * 70)
    metrics = ["tool_recall", "tool_precision", "keyword_hit", "rewrite_keyword_hit"]
    for m in metrics:
        vals = [r[m] for r in results if r[m] is not None]
        if vals:
            print(f"  avg {m}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    parallel_cases = [r for r in results if r["parallel_ok"] is not None]
    if parallel_cases:
        ok = sum(1 for r in parallel_cases if r["parallel_ok"])
        print(f"  parallel_ok: {ok}/{len(parallel_cases)}")
    print("=" * 70)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run golden QA eval")
    parser.add_argument("--ids", help="Comma-separated case IDs to run, e.g. q001,q004")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "eval_dataset.json"))
    parser.add_argument("--out", help="Write JSON results to file")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.ids:
        ids = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in ids]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    print(f"[eval] Loading RAG contexts...")
    rag_contexts = load_rag_contexts()
    workflow = build_workflow(rag_contexts)

    print(f"[eval] Running {len(cases)} cases...\n")
    results = []
    for case in cases:
        print(f"  Running {case['id']}: {case['input'][:60]}...")
        r = eval_case(case, workflow)
        results.append(r)

    print_report(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[eval] Results written to {args.out}")


if __name__ == "__main__":
    main()
