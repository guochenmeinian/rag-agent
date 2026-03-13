"""Retrieval layer evaluator.

Metrics
───────
hit@k   (0/1) — does at least one relevant chunk appear in top-k results?
MRR         — mean reciprocal rank of the first relevant chunk

Input: a ranked list of retrieved chunk IDs and the ground truth relevant set.
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
    """
    gt = case.get("retrieval_gt", {})
    relevant = set(gt.get("relevant_chunk_ids", []))
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
    valid = [r for r in results if not r.get("skip")]
    if not valid:
        return {"n": 0, "skipped": len(results)}

    # Collect all k values seen
    all_k = set()
    for r in valid:
        all_k.update(int(k.split("@")[1]) for k in r["metrics"] if k.startswith("hit@"))

    agg: dict = {"n": len(valid), "skipped": len(results) - len(valid)}
    for k in sorted(all_k):
        key = f"hit@{k}"
        vals = [r["metrics"].get(key, 0) for r in valid]
        agg[key] = round(sum(vals) / len(vals), 3)

    mrr_vals = [r["metrics"]["mrr"] for r in valid]
    agg["mrr"] = round(sum(mrr_vals) / len(mrr_vals), 4)

    return agg
