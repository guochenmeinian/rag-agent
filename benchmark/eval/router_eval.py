"""Router layer evaluator.

Metrics
───────
tool_classification_accuracy  (0/1)
  — Did the model call the right set of tools?
  — Did it skip tools it shouldn't use?
  hard: expected_tools ⊆ called, forbidden_tools ∩ called = ∅
        if no_tool_needed: called must be empty

parameter  (0/1) × 3 sub-checks
  1. correct_tool  — at least one call matches an expected tool name
  2. correct_format — call arguments are valid JSON with expected keys
  3. correct_content — car_model matches, query_keywords appear in query arg

multi_query  (0/1) × 2 sub-checks
  1. efficient    — total calls ≤ max_calls AND no duplicate (name, car_model, query) calls
  2. complete     — total calls ≥ min_calls (all required tools called)
  + parallelism   — if must_be_parallel: all expected tools in one batch
"""
from __future__ import annotations

import re
from collections import Counter


# ─────────────────────────────────────────────────────────────
# Types (mirroring what workflow.run_stream yields)
# ─────────────────────────────────────────────────────────────
# actual_calls: list of {name, **kwargs} dicts from tool_calling events
# batches:      list of lists — each inner list = one parallel batch


# ─────────────────────────────────────────────────────────────
# tool_classification_accuracy
# ─────────────────────────────────────────────────────────────

def score_tool_classification(
    actual_calls: list[dict],
    gt: dict,
) -> dict:
    called_names = [c["name"] for c in actual_calls]
    expected     = gt.get("expected_tools", [])
    forbidden    = gt.get("forbidden_tools", [])
    no_tool      = gt.get("no_tool_needed", False)

    failures = []

    if no_tool:
        if called_names:
            failures.append(f"Expected no tools, but called: {called_names}")
    else:
        # Every expected tool must have been called at least once
        for t in expected:
            if t not in called_names:
                failures.append(f"Expected tool '{t}' was not called")

    # Forbidden tools must not be called
    hit_forbidden = [t for t in forbidden if t in called_names]
    if hit_forbidden:
        failures.append(f"Forbidden tools called: {hit_forbidden}")

    return {"score": int(len(failures) == 0), "failures": failures}


# ─────────────────────────────────────────────────────────────
# parameter (3 sub-checks)
# ─────────────────────────────────────────────────────────────

def _check_correct_tool(actual_calls: list[dict], gt: dict) -> dict:
    """Sub-check 1: at least one call matches each expected tool name."""
    expected = gt.get("expected_tools", [])
    if not expected:
        return {"pass": True, "detail": "no expected tools"}
    called_names = [c["name"] for c in actual_calls]
    missing = [t for t in expected if t not in called_names]
    return {"pass": len(missing) == 0, "missing": missing}


def _check_correct_format(actual_calls: list[dict], gt: dict) -> dict:
    """Sub-check 2: each call to an expected tool has the right argument keys."""
    tool_params = gt.get("tool_params", {})
    failures = []
    for call in actual_calls:
        name = call.get("name")
        if name not in tool_params:
            continue
        expected_params = tool_params[name]
        for key in expected_params:
            if key == "query_keywords":
                # query_keywords is checked in content, not format
                continue
            if key not in call and key != "dense_weight":
                # dense_weight is optional
                failures.append(f"{name}: missing argument '{key}'")
    return {"pass": len(failures) == 0, "failures": failures}


def _check_correct_content(actual_calls: list[dict], gt: dict) -> dict:
    """Sub-check 3: car_model matches, query_keywords appear in query arg."""
    tool_params = gt.get("tool_params", {})
    failures = []

    for call in actual_calls:
        name = call.get("name")
        if name not in tool_params:
            continue
        params = tool_params[name]

        # car_model exact match
        expected_model = params.get("car_model")
        if expected_model:
            actual_model = call.get("car_model", call.get("input", {}).get("car_model", ""))
            if actual_model != expected_model:
                failures.append(f"{name}: car_model expected '{expected_model}', got '{actual_model}'")

        # query_keywords: at least one should appear in the query argument
        # grep_search uses "keywords", rag_search/web_search use "query"
        kws = params.get("query_keywords", [])
        if kws:
            query_arg = (
                call.get("query")
                or call.get("keywords")
                or call.get("input", {}).get("query", "")
                or call.get("input", {}).get("keywords", "")
                or ""
            )
            found = [kw for kw in kws if kw.lower() in query_arg.lower()]
            if not found:
                failures.append(f"{name}: none of {kws} found in query arg '{query_arg[:60]}'")

        # dense_weight range (optional)
        weight_range = params.get("dense_weight")
        if weight_range and isinstance(weight_range, (list, tuple)):
            lo, hi = weight_range
            actual_w = call.get("dense_weight", call.get("input", {}).get("dense_weight"))
            if actual_w is not None and not (lo <= actual_w <= hi):
                failures.append(f"{name}: dense_weight {actual_w} outside [{lo}, {hi}]")

    return {"pass": len(failures) == 0, "failures": failures}


def score_parameter(actual_calls: list[dict], gt: dict) -> dict:
    c1 = _check_correct_tool(actual_calls, gt)
    c2 = _check_correct_format(actual_calls, gt)
    c3 = _check_correct_content(actual_calls, gt)

    # Score = fraction of sub-checks passed (0, 0.33, 0.67, or 1.0 → rounded to 0/1 per sub)
    sub_scores = {
        "correct_tool":    int(c1["pass"]),
        "correct_format":  int(c2["pass"]),
        "correct_content": int(c3["pass"]),
    }
    overall = int(all(v == 1 for v in sub_scores.values()))

    return {
        "score": overall,
        "sub_scores": sub_scores,
        "detail": {"correct_tool": c1, "correct_format": c2, "correct_content": c3},
    }


# ─────────────────────────────────────────────────────────────
# multi_query (2 sub-checks + parallelism)
# ─────────────────────────────────────────────────────────────

def _call_signature(call: dict) -> tuple:
    """Normalized identity tuple for detecting semantically redundant calls."""
    query = re.sub(r"\s+", "", str(
        call.get("query") or call.get("keywords") or call.get("query_keywords") or ""
    )).lower()
    return (call.get("name", ""), call.get("car_model", ""), query)


def score_multi_query(actual_calls: list[dict], batches: list[list[dict]], gt: dict) -> dict:
    n_calls = len(actual_calls)
    min_c   = gt.get("min_calls", 1)
    max_c   = gt.get("max_calls", 1)
    must_parallel = gt.get("must_be_parallel", False)

    # Redundant call detection: same (name, car_model, normalized_query)
    sig_counts = Counter(_call_signature(c) for c in actual_calls)
    dup_count  = sum(v - 1 for v in sig_counts.values() if v > 1)
    no_redundant = dup_count == 0

    # Sub-check 1: efficient (≤ max_calls AND no redundant calls)
    efficient = n_calls <= max_c and no_redundant

    # Sub-check 2: complete (≥ min_calls)
    complete = n_calls >= min_c

    # Parallelism: any single batch with ≥2 calls
    max_batch_size = max((len(b) for b in batches), default=0)
    parallel_ok = (not must_parallel) or (max_batch_size >= 2)

    return {
        "score": int(efficient and complete and parallel_ok),
        "sub_scores": {
            "efficient":   int(efficient),
            "complete":    int(complete),
            "parallel_ok": int(parallel_ok),
        },
        "detail": {
            "n_calls": n_calls,
            "min_calls": min_c,
            "max_calls": max_c,
            "duplicate_call_count": dup_count,
            "max_batch_size": max_batch_size,
            "must_be_parallel": must_parallel,
        },
    }


# ─────────────────────────────────────────────────────────────
# Case-level runner
# ─────────────────────────────────────────────────────────────

def eval_router_case(
    case: dict,
    actual_calls: list[dict],
    batches: list[list[dict]],
) -> dict:
    """Evaluate one router case.

    Args:
        case:         BenchmarkCase dict with router_gt
        actual_calls: flat list of all tool call dicts from the run
        batches:      list of parallel-batch lists (each inner list = one LLM response's calls)
    """
    gt = case.get("router_gt", {})

    tc  = score_tool_classification(actual_calls, gt)
    par = score_parameter(actual_calls, gt)
    mq  = score_multi_query(actual_calls, batches, gt)

    return {
        "id":    case["id"],
        "input": case["input"],
        "metrics": {
            "tool_classification_accuracy": tc["score"],
            "parameter":                    par["score"],
            "multi_query":                  mq["score"],
        },
        "detail": {
            "tool_classification": tc,
            "parameter":           par,
            "multi_query":         mq,
        },
        "actual_calls": [{"name": c["name"], **{k: v for k, v in c.items() if k != "name"}} for c in actual_calls],
    }


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_router(results: list[dict]) -> dict:
    metric_names = ["tool_classification_accuracy", "parameter", "multi_query"]
    agg: dict = {}
    for m in metric_names:
        vals = [r["metrics"][m] for r in results]
        agg[m] = round(sum(vals) / len(vals), 3) if vals else 0.0

    # Sub-score averages (parameter + multi_query)
    par_subs = ["correct_tool", "correct_format", "correct_content"]
    mq_subs  = ["efficient", "complete", "parallel_ok"]

    agg["parameter_sub"] = {
        s: round(
            sum(r["detail"]["parameter"]["sub_scores"].get(s, 0) for r in results) / len(results), 3
        )
        for s in par_subs
    } if results else {}

    agg["multi_query_sub"] = {
        s: round(
            sum(r["detail"]["multi_query"]["sub_scores"].get(s, 0) for r in results) / len(results), 3
        )
        for s in mq_subs
    } if results else {}

    # Diagnostic: average redundant calls per case
    agg["avg_duplicate_calls"] = round(
        sum(r["detail"]["multi_query"]["detail"].get("duplicate_call_count", 0) for r in results) / len(results), 3
    ) if results else 0.0

    return {"n": len(results), **agg}
