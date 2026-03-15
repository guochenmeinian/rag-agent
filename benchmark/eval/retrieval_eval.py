"""Retrieval layer evaluator.

Metrics
───────
Mode A — chunk-ID based (requires offline annotation):
    hit@k        (0/1) — does at least one relevant chunk appear in top-k results?
    MRR              — mean reciprocal rank of the first relevant chunk

Mode B — LLM-judge based (annotation-free, temporary fallback):
    relevance@k  (0/1) — LLM judges whether top-k chunks are relevant to the query intent
    facts_coverage     — fraction of expected_facts found in top-k chunk texts (hard check)

Shared:
    no_hit_ok    (0/1) — for expect_no_hit cases: 1 if retriever returns 0 chunks, 0 otherwise

Input: ranked_chunks — list of {"id": str, "content": str}, ordered most→least relevant.
For Mode A, only "id" is required. For Mode B, "content" is also required.

expect_no_hit
─────────────
Set retrieval_gt.expect_no_hit = true for out-of-domain queries where the retriever
should return nothing (score_threshold filters all chunks).
These cases are excluded from hit@k / MRR / relevance aggregation and tracked separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from benchmark.eval.llm_judge import judge_retrieval_relevance


# ─────────────────────────────────────────────────────────────
# Mode A helpers
# ─────────────────────────────────────────────────────────────

def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> int:
    """Return 1 if any relevant chunk appears in the top-k results."""
    return int(any(cid in relevant_ids for cid in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Return 1/rank of the first relevant chunk, or 0 if none found."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────
# Mode B helpers
# ─────────────────────────────────────────────────────────────

def _facts_coverage(chunks: list[dict], expected_facts: list[str]) -> dict:
    if not expected_facts:
        return {"coverage": 1.0, "missing": []}
    combined = " ".join(c.get("content", "") for c in chunks).lower()
    missing = [f for f in expected_facts if f.lower() not in combined]
    coverage = round((len(expected_facts) - len(missing)) / len(expected_facts), 3)
    return {"coverage": coverage, "missing": missing}


# ─────────────────────────────────────────────────────────────
# Case-level runner
# ─────────────────────────────────────────────────────────────

def eval_retrieval_case(
    case: dict,
    ranked_chunks: list[dict],
) -> dict:
    """Evaluate one retrieval case.

    Args:
        case:          BenchmarkCase dict with retrieval_gt
        ranked_chunks: ordered list of {"id": str, "content": str}
                       (index 0 = most relevant; content required for Mode B)

    Returns:
        Dict with metrics and debug detail.
        For expect_no_hit cases: only no_hit_ok metric is set.
        For Mode A: hit@k + MRR.
        For Mode B: relevance@k + facts_coverage.
        Skipped if neither mode is applicable.
    """
    gt = case.get("retrieval_gt", {})
    expect_no_hit = gt.get("expect_no_hit", False)
    ranked_ids = [c["id"] for c in ranked_chunks]

    # ── expect_no_hit branch ──────────────────────────────────────
    if expect_no_hit:
        return {
            "id":            case["id"],
            "input":         case["input"],
            "skip":          False,
            "expect_no_hit": True,
            "metrics":       {"no_hit_ok": int(len(ranked_chunks) == 0)},
            "detail": {
                "n_chunks_returned": len(ranked_chunks),
                "ranked_top5":       ranked_ids[:5],
            },
        }

    eval_at_k = gt.get("eval_at_k", [5])

    # ── Mode A: chunk-ID based ────────────────────────────────────
    relevant = set(gt.get("relevant_chunk_ids", []))
    if relevant:
        hits = {f"hit@{k}": hit_at_k(ranked_ids, relevant, k) for k in eval_at_k}
        mrr  = reciprocal_rank(ranked_ids, relevant)
        return {
            "id":    case["id"],
            "input": case["input"],
            "skip":  False,
            "mode":  "chunk_id",
            "metrics": {**hits, "mrr": round(mrr, 4)},
            "detail": {
                "ranked_top5": ranked_ids[:5],
                "relevant":    list(relevant),
                "first_hit_rank": next(
                    (i + 1 for i, cid in enumerate(ranked_ids) if cid in relevant),
                    None,
                ),
            },
        }

    # ── Mode B: LLM-judge based ───────────────────────────────────
    query_intent = gt.get("query_intent", "")
    if query_intent:
        k = eval_at_k[0] if eval_at_k else 5
        top_chunks = ranked_chunks[:k]
        llm = judge_retrieval_relevance(case["input"], top_chunks, gt)
        facts = _facts_coverage(top_chunks, gt.get("expected_facts", []))
        return {
            "id":    case["id"],
            "input": case["input"],
            "skip":  False,
            "mode":  "llm_judge",
            "metrics": {
                f"relevance@{k}": llm.get("score", 0),
                "facts_coverage": facts["coverage"],
            },
            "detail": {
                "ranked_top5":  ranked_ids[:5],
                "llm":          llm,
                "facts":        facts,
            },
        }

    # ── Neither mode applicable ───────────────────────────────────
    return {
        "id":    case["id"],
        "skip":  True,
        "reason": "No relevant_chunk_ids or query_intent in retrieval_gt",
    }


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_retrieval(results: list[dict]) -> dict:
    no_hit_cases = [r for r in results if r.get("expect_no_hit")]
    normal_cases = [r for r in results if not r.get("expect_no_hit")]

    valid    = [r for r in normal_cases if not r.get("skip")]
    skipped  = len(normal_cases) - len(valid)

    agg: dict = {"n": len(valid), "skipped": skipped}

    # Mode A aggregation
    mode_a = [r for r in valid if r.get("mode") == "chunk_id"]
    if mode_a:
        all_k = set()
        for r in mode_a:
            all_k.update(int(k.split("@")[1]) for k in r["metrics"] if k.startswith("hit@"))
        for k in sorted(all_k):
            key = f"hit@{k}"
            vals = [r["metrics"].get(key, 0) for r in mode_a]
            agg[key] = round(sum(vals) / len(vals), 3)
        mrr_vals = [r["metrics"]["mrr"] for r in mode_a]
        agg["mrr"] = round(sum(mrr_vals) / len(mrr_vals), 4)
        agg["n_chunk_id"] = len(mode_a)

    # Mode B aggregation
    mode_b = [r for r in valid if r.get("mode") == "llm_judge"]
    if mode_b:
        rel_keys = set()
        for r in mode_b:
            rel_keys.update(k for k in r["metrics"] if k.startswith("relevance@"))
        for key in rel_keys:
            vals = [r["metrics"].get(key, 0) for r in mode_b]
            agg[key] = round(sum(vals) / len(vals), 3)
        fc_vals = [r["metrics"]["facts_coverage"] for r in mode_b]
        agg["facts_coverage_avg"] = round(sum(fc_vals) / len(fc_vals), 3)
        agg["n_llm_judge"] = len(mode_b)

    # expect_no_hit aggregation
    if no_hit_cases:
        no_hit_scores = [r["metrics"]["no_hit_ok"] for r in no_hit_cases]
        agg["no_hit_n"]  = len(no_hit_cases)
        agg["no_hit_ok"] = round(sum(no_hit_scores) / len(no_hit_scores), 3)

    return agg
