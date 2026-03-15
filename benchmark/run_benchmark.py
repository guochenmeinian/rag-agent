"""Benchmark runner — evaluates the full RAG agent pipeline layer by layer.

Usage
─────
# Run all layers on all synthesized cases
python benchmark/run_benchmark.py

# Only specific layers
python benchmark/run_benchmark.py --layers rewriter router

# Filter by case ID or category (= synthesis target)
python benchmark/run_benchmark.py --ids rw_standalone_coref_001,rw_entity_001
python benchmark/run_benchmark.py --category rw_standalone_coref

# Custom dataset / output
python benchmark/run_benchmark.py --dataset benchmark/data/cases.json --out benchmark/data/results/run_01.json

# Dry-run: load cases, skip all LLM/workflow calls
python benchmark/run_benchmark.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import config as src_config
from agent.workflow import AgentWorkflow

import benchmark.config as bm_config
from benchmark.eval.rewriter_eval import eval_rewriter_case, aggregate_rewriter
from benchmark.eval.router_eval import eval_router_case, aggregate_router
from benchmark.eval.retrieval_eval import eval_retrieval_case, aggregate_retrieval
from benchmark.eval.answer_eval import eval_answer_case, aggregate_answer

if TYPE_CHECKING:
    from rag.pipeline import RAGContext

ALL_LAYERS = ["rewriter", "router", "retrieval", "answer"]


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


# ─────────────────────────────────────────────────────────────
# Workflow factory
# ─────────────────────────────────────────────────────────────

def load_rag_contexts() -> dict[str, "RAGContext"]:
    from rag.pipeline import ingest

    contexts: dict[str, "RAGContext"] = {}
    for model in src_config.NIO_CAR_MODELS:
        subdir = os.path.join(src_config.DATA_ROOT, model)
        flat_pdf = os.path.join(src_config.DATA_ROOT, f"{model}.pdf")
        col_name = f"nio_{model.lower()}"
        try:
            if os.path.isdir(subdir):
                ctx = ingest(data_dir=subdir, uri=src_config.MILVUS_URI, col_name=col_name)
            elif os.path.isfile(flat_pdf):
                ctx = ingest(
                    data_dir=src_config.DATA_ROOT,
                    uri=src_config.MILVUS_URI,
                    col_name=col_name,
                    file_filter=f"{model}.pdf",
                )
            else:
                continue
            contexts[model] = ctx
        except Exception as exc:
            print(f"[warn] failed to load RAG context for {model}: {exc}")
    return contexts


def make_workflow(
    rag_contexts: dict[str, "RAGContext"] | None = None,
    disabled: set[str] | None = None,
) -> AgentWorkflow:
    from tools.registry import ToolRegistry
    from tools.rag_search import RagSearchTool
    from tools.grep_search import GrepSearchTool
    from tools.web_search import WebSearchTool

    disabled = disabled or set()
    registry = ToolRegistry()
    if "rag" not in disabled:
        registry.register(RagSearchTool(contexts=rag_contexts or {}))
        registry.register(GrepSearchTool())
    if "web" not in disabled:
        registry.register(WebSearchTool())
    return AgentWorkflow(registry=registry, disabled=disabled)


def inject_context(workflow: AgentWorkflow, ctx: dict | None):
    if not ctx:
        return
    if ctx.get("user_profile"):
        workflow.memory.global_user_info.raw = ctx["user_profile"]
    for fact in ctx.get("memory_facts", []):
        workflow.memory.facts.append(fact)
    for msg in ctx.get("conversation_history", []):
        workflow.memory.add_message(msg["role"], msg["content"])


# ─────────────────────────────────────────────────────────────
# Event stream parsing
# ─────────────────────────────────────────────────────────────

def parse_events(events: list[dict]) -> dict:
    """Unpack all event types into a structured result dict.

    rewrite_result — {"type": "rewrite"|"clarify", "content": str}
    actual_calls   — flat list of all tool call dicts
    batches        — one inner list per LLM response (for parallel detection)
    answer         — final answer string
    tool_results   — raw tool result objects
    """
    rewrite_result: dict = {"type": "rewrite", "content": ""}
    actual_calls:   list[dict] = []
    batches:        list[list] = []
    answer:         str = ""
    tool_results:   list[dict] = []
    usage:          dict = {}

    for ev in events:
        t = ev["type"]
        if t == "clarify":
            rewrite_result = {"type": "clarify", "content": ev.get("message", "")}
        elif t == "refined":
            rewrite_result = {"type": "rewrite", "content": ev.get("query", "")}
        elif t == "tool_calling":
            batch = ev.get("calls", [])
            actual_calls.extend(batch)
            batches.append(list(batch))
        elif t == "tool_done":
            tool_results.extend(ev.get("results", []))
        elif t == "done":
            answer = ev.get("answer", "")
            usage  = ev.get("usage", {})

    return {
        "rewrite_result": rewrite_result,
        "actual_calls":   actual_calls,
        "batches":        batches,
        "answer":         answer,
        "tool_results":   tool_results,
        "usage":          usage,
    }


def _extract_chunks(tool_results: list[dict]) -> list[dict]:
    """Extract retrieved content as [{"id": str, "content": str}] from tool results.

    rag_search stores formatted citations in ToolResult.content (not individual chunk IDs),
    so each tool call becomes one entry keyed by tool_name + car_model.
    """
    chunks_out: list[dict] = []
    for tr in tool_results:
        result_obj = tr.get("result")
        if result_obj is None or not getattr(result_obj, "success", True):
            continue
        content = getattr(result_obj, "content", "") or ""
        meta    = getattr(result_obj, "metadata", {}) or {}
        cid = f"{tr.get('name', 'tool')}:{meta.get('car_model', meta.get('query', '?'))}"
        if content:
            chunks_out.append({"id": cid, "content": content})
    return chunks_out


# ─────────────────────────────────────────────────────────────
# Per-case runner
# ─────────────────────────────────────────────────────────────

def run_case(
    case: dict,
    layers: list[str],
    dry_run: bool = False,
    rag_contexts: dict[str, "RAGContext"] | None = None,
    disabled: set[str] | None = None,
) -> dict:
    if dry_run:
        return {"id": case["id"], "dry_run": True, "input": case["input"]}

    disabled = disabled or set()
    targets = case.get("layer_targets", [])
    workflow = make_workflow(rag_contexts=rag_contexts, disabled=disabled)
    if "memory" not in disabled:
        inject_context(workflow, case.get("context"))

    events: list[dict] = []
    t0 = time.monotonic()
    try:
        for ev in workflow.run_stream(case["input"]):
            events.append(ev)
    except Exception as exc:
        return {"id": case["id"], "error": str(exc), "input": case["input"]}

    latency = round(time.monotonic() - t0, 2)
    parsed  = parse_events(events)

    result: dict = {
        "id":        case["id"],
        "category":  case.get("category", ""),
        "input":     case["input"],
        "latency_s": latency,
        "usage":     parsed.get("usage", {}),
        "metrics":   {},
        "detail":    {},
    }

    if "rewriter" in layers and "rewriter" in targets and case.get("rewriter_gt"):
        if "rewriter" in disabled:
            result["detail"]["rewriter"] = {"skipped": True, "reason": "rewriter disabled"}
        else:
            r = eval_rewriter_case(case, parsed["rewrite_result"])
            result["metrics"].update({f"rewriter/{k}": v for k, v in r["metrics"].items()})
            result["detail"]["rewriter"] = r["detail"]
            result["rewrite_result"] = parsed["rewrite_result"]

    if "router" in layers and "router" in targets and case.get("router_gt"):
        r = eval_router_case(case, parsed["actual_calls"], parsed["batches"])
        result["metrics"].update({f"router/{k}": v for k, v in r["metrics"].items()})
        result["detail"]["router"] = r["detail"]
        result["actual_calls"] = parsed["actual_calls"]

    if "retrieval" in layers and "retrieval" in targets and case.get("retrieval_gt"):
        ranked_chunks = _extract_chunks(parsed["tool_results"])
        r = eval_retrieval_case(case, ranked_chunks)
        if not r.get("skip"):
            result["metrics"].update({f"retrieval/{k}": v for k, v in r["metrics"].items()})
            result["detail"]["retrieval"] = {**r.get("detail", {}), "mode": r.get("mode")}
            if r.get("expect_no_hit"):
                result["detail"]["retrieval"]["expect_no_hit"] = True

    if "answer" in layers and "answer" in targets and case.get("answer_gt"):
        r = eval_answer_case(case, parsed["answer"])
        result["metrics"].update({f"answer/{k}": v for k, v in r["metrics"].items()})
        result["detail"]["answer"] = r["detail"]
        result["answer"] = parsed["answer"][:300]

    return result


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def _strip_prefix(metrics: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in metrics.items() if k.startswith(prefix)}


def aggregate_all(results: list[dict], layers: list[str]) -> dict:
    valid  = [r for r in results if not r.get("error") and not r.get("dry_run")]
    summary: dict = {
        "n_total": len(results),
        "n_run":   len(valid),
        "n_error": sum(1 for r in results if r.get("error")),
    }

    for layer in layers:
        prefix = f"{layer}/"
        layer_results = [r for r in valid if any(k.startswith(prefix) for k in r["metrics"])]
        if not layer_results:
            continue

        flat = [{"id": r["id"], "category": r["category"],
                 "metrics": _strip_prefix(r["metrics"], prefix),
                 "detail":  r.get("detail", {}).get(layer, {})}
                for r in layer_results]

        if layer == "rewriter":
            summary["rewriter"] = aggregate_rewriter(flat)
        elif layer == "router":
            summary["router"] = aggregate_router(flat)
        elif layer == "retrieval":
            ret_detail = lambda r: r.get("detail", {}).get("retrieval", {})
            full = [{"id": r["id"], "input": r["input"], "skip": False,
                     "metrics":      _strip_prefix(r["metrics"], prefix),
                     "detail":       ret_detail(r),
                     "mode":         ret_detail(r).get("mode"),
                     "expect_no_hit": ret_detail(r).get("expect_no_hit", False)}
                    for r in layer_results]
            summary["retrieval"] = aggregate_retrieval(full)
        elif layer == "answer":
            summary["answer"] = aggregate_answer(flat)

    latencies = [r["latency_s"] for r in valid if "latency_s" in r]
    if latencies:
        s = sorted(latencies)
        summary["latency"] = {
            "p50": s[int(len(s) * 0.5)],
            "p95": s[min(int(len(s) * 0.95), len(s) - 1)],
        }

    summary["cost"] = {
        "total_prompt_tokens":     sum(r.get("usage", {}).get("prompt_tokens", 0) for r in valid),
        "total_completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in valid),
        "total_tokens":            sum(r.get("usage", {}).get("total_tokens", 0) for r in valid),
    }

    return summary


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_summary(summary: dict):
    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  Cases: {summary['n_run']} run / {summary.get('n_error', 0)} errors\n")

    LAYER_METRICS = {
        "rewriter":  [
            ("standalone",                 "standalone_n"),
            ("entity_extraction_accuracy", "entity_n"),
            ("clarify_detection",          "clarify_n"),
            ("coref_resolution_rate",      None),
            ("ellipsis_fill_rate",         None),
        ],
        "router":    [
            ("tool_classification_accuracy", None),
            ("parameter",                    None),
            ("multi_query",                  None),
            ("avg_duplicate_calls",          None),
        ],
        "retrieval": [
            ("hit@1", None), ("hit@3", None), ("hit@5", None), ("mrr", "n_chunk_id"),
            ("relevance@5", "n_llm_judge"), ("facts_coverage_avg", None),
            ("no_hit_ok", "no_hit_n"),
        ],
        "answer":    [
            ("match_avg",               None),
            ("match_full",              None),
            ("hallucination_clean",     None),
            ("clarification_acc",       None),
            ("key_facts_coverage_avg",  None),
        ],
    }

    for layer, metrics in LAYER_METRICS.items():
        if layer not in summary:
            continue
        data = summary[layer]
        print(f"  {layer.upper()}  (n={data.get('n', '?')})")
        for m, n_key in metrics:
            if m not in data:
                continue
            n_str = f"  n={data[n_key]}" if n_key and n_key in data else ""
            val = data[m]
            val_str = "N/A" if val is None else f"{val:.3f}"
            print(f"    {m:<35} {val_str}{n_str}")
        print()

    if "latency" in summary:
        lat = summary["latency"]
        print(f"  LATENCY  p50={lat['p50']}s  p95={lat['p95']}s")

    if "cost" in summary:
        c = summary["cost"]
        print(f"  COST     prompt={c['total_prompt_tokens']}  completion={c['total_completion_tokens']}  total={c['total_tokens']} tokens")

    print("=" * 65 + "\n")


def print_case_row(r: dict):
    if r.get("dry_run"):
        print(f"  [DRY ] {r['id']}")
        return
    if r.get("error"):
        print(f"  [ERR ] {r['id']} — {r['error'][:60]}")
        return
    metrics_str = "  ".join(
        f"{k.split('/')[1][:4]}={'N/A' if v is None else v}"
        for k, v in list(r.get("metrics", {}).items())[:4]
    )
    print(f"  [OK  ] {r['id']:<32} {metrics_str}  {r.get('latency_s', '?')}s")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

DISABLEABLE = ["rewriter", "rag", "web", "memory"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run layered RAG agent benchmark")
    p.add_argument("--dataset",  default=str(bm_config.CASES_FILE))
    p.add_argument("--layers",   nargs="+", default=ALL_LAYERS, choices=ALL_LAYERS)
    p.add_argument("--ids",      default="", help="Comma-separated case IDs")
    p.add_argument("--category", default="", help="Filter by category (= synthesis target id)")
    p.add_argument("--out",      default="", help="Write JSON results to file")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--disable",  nargs="+", default=[], choices=DISABLEABLE,
                   metavar="COMPONENT",
                   help=f"Disable pipeline components for ablation. Choices: {DISABLEABLE}")
    return p.parse_args()


def main():
    args = parse_args()
    disabled = set(args.disable)
    cases = load_dataset(args.dataset)
    rag_contexts = None if args.dry_run else load_rag_contexts()
    if not args.dry_run and not rag_contexts:
        print("[warn] no RAG contexts loaded; rag_search will return not_found for all models")

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]

    if not cases:
        print("No cases matched the filter.")
        return

    disabled_str = f" | disabled: {sorted(disabled)}" if disabled else ""
    print(f"\nRunning {len(cases)} cases | layers: {args.layers}{disabled_str}\n")

    results = []
    for case in cases:
        print(f"  → {case['id']} : {case['input'][:55]}")
        r = run_case(case, layers=args.layers, dry_run=args.dry_run,
                     rag_contexts=rag_contexts, disabled=disabled)
        results.append(r)
        print_case_row(r)

    summary = aggregate_all(results, args.layers)
    print_summary(summary)

    out_path = args.out or str(bm_config.RESULTS_DIR / f"run_{int(time.time())}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
