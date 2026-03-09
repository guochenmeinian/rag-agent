"""Benchmark: tool-selection accuracy against eval_dataset.json.

Measures how well the AgentWorkflow (GPT-4o executor + Qwen rewriter) selects the
right tools, constructs valid arguments, and issues parallel calls when needed.

Usage:
    cd src && python tests/benchmark_tool_selection.py [OPTIONS]

Options:
    --ids       q001,q004,q011   Run specific case IDs (comma-separated)
    --category  multi_car_compare Run all cases in a category
    --dataset   path/to/eval_dataset.json  Override dataset path
    --out       results.json      Write JSON results to file
    --no-rag    Skip cases requiring RAG (faster, no Milvus needed)
    --model     gpt-4o           Override executor model (env var takes precedence)

Metrics (per case, then aggregated):
    tool_name_acc       Whether correct tool names were called (precision+recall F1)
    car_model_acc       Whether correct car_model args were passed
    parallel_ok         Whether ≥2 tools issued in a single response when expected
    rewrite_hit         Whether refined query contains expected keywords (multi-turn cases)
    keyword_hit         Whether final answer contains expected_answer_keywords
    hallucination_miss  Whether answer avoids must_not_contain strings
    latency_s           Wall-clock seconds for the full run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Path bootstrap ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from agent.memory import ConversationMemory
from agent.workflow import AgentWorkflow
from tools.registry import ToolRegistry
from tools.rag_search import RagSearchTool
from tools.grep_search import GrepSearchTool
from tools.web_search import WebSearchTool


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────

DEFAULT_DATASET = Path(__file__).parent / "eval_dataset.json"


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


# ─────────────────────────────────────────────────────────────
# Workflow factory
# ─────────────────────────────────────────────────────────────

def make_workflow(contexts: dict | None = None) -> AgentWorkflow:
    """Build a fresh workflow per case (no cross-case memory leak)."""
    registry = ToolRegistry()
    registry.register(RagSearchTool(contexts=contexts or {}))
    registry.register(GrepSearchTool())
    registry.register(WebSearchTool())
    return AgentWorkflow(registry=registry)


def inject_context(workflow: AgentWorkflow, ctx: dict | None):
    """Pre-load memory with the context dict from eval_dataset case."""
    if not ctx:
        return
    if summary := ctx.get("summary"):
        workflow.memory.global_user_info.raw = summary
    if "focus_models" in ctx:
        workflow.memory.global_user_info.focus_models = ctx["focus_models"]
    if "facts" in ctx:
        workflow.memory.facts = ctx["facts"]
    for msg in ctx.get("recent_messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        workflow.memory.add_message(role, content)


# ─────────────────────────────────────────────────────────────
# Event parsing helpers
# ─────────────────────────────────────────────────────────────

def extract_tool_calls_from_events(events: list[dict]) -> list[dict]:
    """Collect all tool_calling events; return flat list of call dicts."""
    calls = []
    for ev in events:
        if ev["type"] == "tool_calling":
            calls.extend(ev.get("calls", []))
    return calls


def was_parallel(events: list[dict]) -> bool:
    """Return True if any single tool_calling event issued ≥2 calls."""
    for ev in events:
        if ev["type"] == "tool_calling" and len(ev.get("calls", [])) >= 2:
            return True
    return False


def get_refined_query(events: list[dict]) -> str:
    for ev in events:
        if ev["type"] == "refined":
            return ev.get("query", "")
    return ""


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

def score_tool_names(
    expected: list[dict],
    actual_calls: list[dict],
) -> dict:
    """F1 over tool names (multi-set precision + recall)."""
    expected_names = [e["name"] for e in expected]
    actual_names   = [c["name"] for c in actual_calls]

    if not expected_names and not actual_names:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected_names:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not actual_names:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    # Multi-set intersection
    from collections import Counter
    exp_ctr = Counter(expected_names)
    act_ctr = Counter(actual_names)
    hits = sum((exp_ctr & act_ctr).values())

    precision = hits / len(actual_names)
    recall    = hits / len(expected_names)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def score_car_models(
    expected: list[dict],
    actual_calls: list[dict],
) -> float:
    """Fraction of expected car_model args that appear in actual calls."""
    expected_models = {e["car_model"] for e in expected if "car_model" in e}
    if not expected_models:
        return 1.0   # no car_model requirement
    actual_models = {c.get("car_model", "") for c in actual_calls}
    hit = len(expected_models & actual_models)
    return round(hit / len(expected_models), 3)


def score_query_keywords(
    expected: list[dict],
    actual_calls: list[dict],
) -> float:
    """Avg fraction of expected query_keywords present in actual query args."""
    scores = []
    for exp in expected:
        kws = exp.get("query_keywords", [])
        if not kws:
            continue
        # Find the matching tool call by name (+ car_model if present)
        matched_queries = [
            c.get("query", "").lower()
            for c in actual_calls
            if c["name"] == exp["name"]
            and (
                "car_model" not in exp
                or c.get("car_model") == exp["car_model"]
            )
        ]
        if not matched_queries:
            scores.append(0.0)
            continue
        best = max(
            sum(kw.lower() in q for kw in kws) / len(kws)
            for q in matched_queries
        )
        scores.append(best)
    return round(sum(scores) / len(scores), 3) if scores else 1.0


def score_answer_keywords(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(kw.lower() in answer_lower for kw in keywords)
    return round(hits / len(keywords), 3)


def score_hallucination(answer: str, must_not: list[str]) -> bool:
    """Return True if answer is clean (no forbidden strings)."""
    answer_lower = answer.lower()
    return all(s.lower() not in answer_lower for s in must_not)


def score_rewrite(refined: str, expected_contains: list[str]) -> float:
    if not expected_contains:
        return 1.0
    refined_lower = refined.lower()
    hits = sum(kw.lower() in refined_lower for kw in expected_contains)
    return round(hits / len(expected_contains), 3)


# ─────────────────────────────────────────────────────────────
# Single case runner
# ─────────────────────────────────────────────────────────────

def run_case(case: dict, skip_rag: bool = False) -> dict:
    """Run one eval case and return a scored result dict."""
    case_id  = case["id"]
    category = case["category"]

    if skip_rag and any(
        e.get("name") == "rag_search"
        for e in case.get("expected_tools", [])
    ):
        return {"id": case_id, "skipped": True, "reason": "rag_search required, --no-rag set"}

    workflow = make_workflow()
    inject_context(workflow, case.get("context"))

    events: list[dict] = []
    answer = ""
    t0 = time.monotonic()

    try:
        for ev in workflow.run_stream(case["input"]):
            events.append(ev)
            if ev["type"] == "done":
                answer = ev.get("answer", "")
    except Exception as exc:
        return {"id": case_id, "error": str(exc), "skipped": False}

    latency = round(time.monotonic() - t0, 2)

    actual_calls   = extract_tool_calls_from_events(events)
    expected_tools = case.get("expected_tools", [])

    tool_name_scores  = score_tool_names(expected_tools, actual_calls)
    car_model_acc     = score_car_models(expected_tools, actual_calls)
    query_kw_acc      = score_query_keywords(expected_tools, actual_calls)
    answer_kw_acc     = score_answer_keywords(answer, case.get("expected_answer_keywords", []))
    hallucination_ok  = score_hallucination(answer, case.get("must_not_contain", []))

    parallel_expected = case.get("expected_parallel", False)
    parallel_actual   = was_parallel(events)
    parallel_ok       = (not parallel_expected) or parallel_actual

    refined = get_refined_query(events)
    rewrite_acc = score_rewrite(refined, case.get("expected_rewrite_contains", []))

    return {
        "id":             case_id,
        "category":       category,
        "skipped":        False,
        "error":          None,
        "latency_s":      latency,
        # Tool selection
        "tool_f1":        tool_name_scores["f1"],
        "tool_precision": tool_name_scores["precision"],
        "tool_recall":    tool_name_scores["recall"],
        "car_model_acc":  car_model_acc,
        "query_kw_acc":   query_kw_acc,
        # Parallel
        "parallel_expected": parallel_expected,
        "parallel_actual":   parallel_actual,
        "parallel_ok":       parallel_ok,
        # Answer quality
        "rewrite_acc":      rewrite_acc,
        "answer_kw_acc":    answer_kw_acc,
        "hallucination_ok": hallucination_ok,
        # Detail for debugging
        "input":          case["input"],
        "refined_query":  refined,
        "answer":         answer[:300],   # truncated for readability
        "actual_tools":   [{"name": c["name"], "car_model": c.get("car_model", ""), "query": c.get("query", "")} for c in actual_calls],
    }


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    valid = [r for r in results if not r.get("skipped") and not r.get("error")]
    if not valid:
        return {}

    def avg(key: str) -> float:
        vals = [r[key] for r in valid if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def frac(key: str) -> float:
        bools = [r[key] for r in valid if isinstance(r.get(key), bool)]
        return round(sum(bools) / len(bools), 3) if bools else 0.0

    by_cat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_cat[r["category"]].append(r)

    cat_summary = {}
    for cat, rows in by_cat.items():
        cat_summary[cat] = {
            "n":            len(rows),
            "tool_f1":      round(sum(r["tool_f1"] for r in rows) / len(rows), 3),
            "parallel_ok":  round(sum(r["parallel_ok"] for r in rows if r["parallel_expected"]) /
                                  max(1, sum(r["parallel_expected"] for r in rows)), 3),
            "answer_kw":    round(sum(r["answer_kw_acc"] for r in rows) / len(rows), 3),
        }

    return {
        "n_total":          len(results),
        "n_run":            len(valid),
        "n_skipped":        sum(1 for r in results if r.get("skipped")),
        "n_error":          sum(1 for r in results if r.get("error")),
        # Core metrics
        "tool_f1_avg":      avg("tool_f1"),
        "tool_precision":   avg("tool_precision"),
        "tool_recall":      avg("tool_recall"),
        "car_model_acc":    avg("car_model_acc"),
        "query_kw_acc":     avg("query_kw_acc"),
        # Parallel
        "parallel_ok_rate": frac("parallel_ok"),
        # Answer quality
        "rewrite_acc":      avg("rewrite_acc"),
        "answer_kw_acc":    avg("answer_kw_acc"),
        "hallucination_clean": frac("hallucination_ok"),
        # Latency
        "latency_p50":      _percentile([r["latency_s"] for r in valid], 50),
        "latency_p95":      _percentile([r["latency_s"] for r in valid], 95),
        # By category
        "by_category":      cat_summary,
    }


def _percentile(vals: list[float], p: int) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return round(s[min(idx, len(s) - 1)], 2)


# ─────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────

def print_summary(agg: dict):
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Cases:         {agg['n_run']} run / {agg['n_skipped']} skipped / {agg['n_error']} errors")
    print()
    print("  TOOL SELECTION")
    print(f"    F1            : {agg['tool_f1_avg']:.3f}")
    print(f"    Precision     : {agg['tool_precision']:.3f}")
    print(f"    Recall        : {agg['tool_recall']:.3f}")
    print(f"    Car model acc : {agg['car_model_acc']:.3f}")
    print(f"    Query kw acc  : {agg['query_kw_acc']:.3f}")
    print()
    print("  PARALLEL CALLS")
    print(f"    Parallel OK   : {agg['parallel_ok_rate']:.3f}")
    print()
    print("  ANSWER QUALITY")
    print(f"    Rewrite acc   : {agg['rewrite_acc']:.3f}")
    print(f"    Answer kw     : {agg['answer_kw_acc']:.3f}")
    print(f"    No hallucin   : {agg['hallucination_clean']:.3f}")
    print()
    print("  LATENCY")
    print(f"    P50           : {agg['latency_p50']}s")
    print(f"    P95           : {agg['latency_p95']}s")
    print()
    print("  BY CATEGORY")
    for cat, cs in agg.get("by_category", {}).items():
        print(f"    {cat:<24} n={cs['n']}  tool_f1={cs['tool_f1']:.2f}  "
              f"parallel_ok={cs['parallel_ok']:.2f}  answer_kw={cs['answer_kw']:.2f}")
    print("=" * 60 + "\n")


def print_case_row(r: dict):
    status = "SKIP" if r.get("skipped") else ("ERR " if r.get("error") else "OK  ")
    if r.get("skipped") or r.get("error"):
        print(f"  [{status}] {r['id']}  {r.get('reason', r.get('error', ''))}")
        return
    parallel_flag = "P" if r["parallel_actual"] else (" " if not r["parallel_expected"] else "!")
    print(
        f"  [{status}] {r['id']:<6} "
        f"f1={r['tool_f1']:.2f} "
        f"car={r['car_model_acc']:.2f} "
        f"par={parallel_flag} "
        f"ans={r['answer_kw_acc']:.2f} "
        f"hal={'✓' if r['hallucination_ok'] else '✗'} "
        f"{r['latency_s']}s  "
        f"{r['input'][:40]}"
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark tool selection accuracy")
    p.add_argument("--ids",      type=str, default="", help="Comma-separated case IDs")
    p.add_argument("--category", type=str, default="", help="Filter by category name")
    p.add_argument("--dataset",  type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--out",      type=str, default="", help="Write JSON results to file")
    p.add_argument("--no-rag",   action="store_true", help="Skip rag_search cases")
    return p.parse_args()


def main():
    args = parse_args()
    cases = load_dataset(args.dataset)

    # Filter
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    if not cases:
        print("No cases matched the filter.")
        sys.exit(0)

    print(f"\nRunning {len(cases)} cases from {args.dataset} …\n")

    results = []
    for case in cases:
        print(f"  → {case['id']} : {case['input'][:60]}")
        r = run_case(case, skip_rag=args.no_rag)
        results.append(r)
        print_case_row(r)

    agg = aggregate(results)
    print_summary(agg)

    if args.out:
        output = {"summary": agg, "results": results}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
