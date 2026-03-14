"""Retrieval layer evaluator.

Metrics
───────
hit@k        (0/1) — does at least one relevant chunk appear in top-k results?
MRR              — mean reciprocal rank of the first relevant chunk
no_hit_ok    (0/1) — for expect_no_hit cases: 1 if retriever returns 0 chunks, 0 otherwise

Input: a ranked list of retrieved chunk IDs and the ground truth relevant set.

expect_no_hit
─────────────
Set retrieval_gt.expect_no_hit = true for out-of-domain queries where the retriever
should return nothing (score_threshold filters all chunks).
These cases are excluded from hit@k / MRR aggregation and tracked separately.
"""
from __future__ import annotations


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> int:
    """Return 1 if any relevant chunk appears in the top-k results."""
    return int(any(cid in relevant_ids for cid in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Return 1/rank of the first relevant chunk, or 0 if none found."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def eval_retrieval_case(
    case: dict,
    ranked_chunk_ids: list[str],
) -> dict:
    """Evaluate one retrieval case.

    Args:
        case:             BenchmarkCase dict with retrieval_gt
        ranked_chunk_ids: ordered list of chunk IDs returned by the retriever
                          (index 0 = most relevant)

    Returns:
        Dict with hit@k scores, MRR, and debug detail.
        For expect_no_hit cases: only no_hit_ok metric is set.
    """
    gt = case.get("retrieval_gt", {})
    expect_no_hit = gt.get("expect_no_hit", False)

    # ── expect_no_hit branch ──────────────────────────────────────
    if expect_no_hit:
        no_hit_ok = int(len(ranked_chunk_ids) == 0)
        return {
            "id":            case["id"],
            "input":         case["input"],
            "skip":          False,
            "expect_no_hit": True,
            "metrics":       {"no_hit_ok": no_hit_ok},
            "detail": {
                "n_chunks_returned": len(ranked_chunk_ids),
                "ranked_top5":       ranked_chunk_ids[:5],
            },
        }

    # ── normal hit@k / MRR branch ─────────────────────────────────
    relevant  = set(gt.get("relevant_chunk_ids", []))
    eval_at_k = gt.get("eval_at_k", [1, 3, 5])

    if not relevant:
        return {
            "id": case["id"],
            "skip": True,
            "reason": "No relevant_chunk_ids annotated (offline annotation pending)",
        }

    hits = {f"hit@{k}": hit_at_k(ranked_chunk_ids, relevant, k) for k in eval_at_k}
    mrr  = reciprocal_rank(ranked_chunk_ids, relevant)

    return {
        "id":     case["id"],
        "input":  case["input"],
        "skip":   False,
        "metrics": {
            **hits,
            "mrr": round(mrr, 4),
        },
        "detail": {
            "ranked_top5": ranked_chunk_ids[:5],
            "relevant":    list(relevant),
            "first_hit_rank": next(
                (i + 1 for i, cid in enumerate(ranked_chunk_ids) if cid in relevant),
                None,
            ),
        },
    }


def aggregate_retrieval(results: list[dict]) -> dict:
    # Separate expect_no_hit cases from normal cases
    no_hit_cases = [r for r in results if r.get("expect_no_hit")]
    normal_cases = [r for r in results if not r.get("expect_no_hit")]

    valid = [r for r in normal_cases if not r.get("skip")]
    skipped = len(normal_cases) - len(valid)

    agg: dict = {"n": len(valid), "skipped": skipped}

    if valid:
        # Collect all k values seen
        all_k = set()
        for r in valid:
            all_k.update(int(k.split("@")[1]) for k in r["metrics"] if k.startswith("hit@"))

        for k in sorted(all_k):
            key = f"hit@{k}"
            vals = [r["metrics"].get(key, 0) for r in valid]
            agg[key] = round(sum(vals) / len(vals), 3)

        mrr_vals = [r["metrics"]["mrr"] for r in valid]
        agg["mrr"] = round(sum(mrr_vals) / len(mrr_vals), 4)

    # expect_no_hit aggregation
    if no_hit_cases:
        no_hit_scores = [r["metrics"]["no_hit_ok"] for r in no_hit_cases]
        agg["no_hit_n"]  = len(no_hit_cases)
        agg["no_hit_ok"] = round(sum(no_hit_scores) / len(no_hit_scores), 3)

    return agg
