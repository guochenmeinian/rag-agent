"""Answer layer evaluator.

Metrics
───────
match          (0/1/2) — similarity to ground truth
  key_facts:  per-fact LLM check — semantic equivalence, not substring match
              "4座" = "4个座位" = "座位数：4"; missing any fact caps score at 1
  semantic_claims: fed into overall LLM judge for holistic scoring
  soft: LLM judge (0=wrong, 1=partial, 2=correct)

hallucination  (0/1) — 1=clean, 0=hallucinated
  hard: forbidden_content substring check (exact wrong values)
  soft: LLM judge

clarification  (0/1) — correct uncertainty / safety handling
  soft: LLM judge only

faithfulness   (0.0–1.0) — RAGAS-style: answer claims supported by retrieved chunks
  vs hallucination: hallucination compares to ground_truth;
                    faithfulness compares to actually retrieved context

context_recall (0.0–1.0) — RAGAS-style: ground truth sentences covered by retrieved chunks
  measures whether retrieval found everything needed to answer correctly
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from benchmark.eval.llm_judge import (
    judge_key_facts,
    judge_answer_match,
    judge_hallucination,
    judge_answer_clarification,
)
from benchmark.eval.ragas_eval import evaluate_ragas


# ─────────────────────────────────────────────────────────────
# Hard check (forbidden content only)
# ─────────────────────────────────────────────────────────────

def _check_forbidden_content(answer: str, forbidden: list[str]) -> dict:
    found = [f for f in forbidden if f.lower() in answer.lower()]
    return {"pass": len(found) == 0, "found": found}


# ─────────────────────────────────────────────────────────────
# Per-metric scorers
# ─────────────────────────────────────────────────────────────

def score_match(answer: str, gt: dict) -> dict:
    """0=wrong, 1=partial, 2=correct.

    key_facts:      per-fact LLM check; any missing → cap final score at 1.
    semantic_claims: passed to overall LLM judge for holistic scoring.
    Final:          min(llm_score, 1) if key_facts incomplete, else llm_score.
    """
    kf  = judge_key_facts(answer, gt.get("key_facts", []))
    llm = judge_answer_match(answer, gt)
    llm_score = llm.get("score", 0)

    if not kf["pass"] and llm_score == 2:
        llm_score = 1

    return {
        "score":     llm_score,
        "key_facts": kf,
        "llm":       llm,
    }


def score_hallucination(answer: str, gt: dict) -> dict:
    """1 = clean, 0 = hallucinated.

    Hard: forbidden_content substring check. Fails → score=0 regardless of LLM.
    Soft: LLM judge.
    """
    fc  = _check_forbidden_content(answer, gt.get("forbidden_content", []))
    llm = judge_hallucination(answer, gt)
    llm_pass = llm.get("score", 1) == 1

    return {
        "score": int(fc["pass"] and llm_pass),
        "hard":  fc,
        "llm":   llm,
    }


def score_clarification(answer: str, gt: dict) -> dict:
    """1 = model correctly expressed uncertainty or refused, 0 = failed."""
    llm = judge_answer_clarification(answer, gt)
    return {
        "score": llm.get("score", 0),
        "llm":   llm,
    }


# ─────────────────────────────────────────────────────────────
# Case-level runner
# ─────────────────────────────────────────────────────────────

def eval_answer_case(case: dict, answer: str, contexts: list[dict] | None = None) -> dict:
    """Evaluate a single answer case.

    Args:
        case:     benchmark case dict with answer_gt
        answer:   system-generated answer string
        contexts: optional list of {"id": str, "content": str} retrieved chunks.
                  When provided, faithfulness and context_recall are computed.
    """
    gt = case.get("answer_gt", {})

    match  = score_match(answer, gt)
    hallu  = score_hallucination(answer, gt)
    clarif = score_clarification(answer, gt)

    metrics = {
        "match":         match["score"],
        "hallucination": hallu["score"],
        "clarification": clarif["score"],
    }
    detail = {
        "match":         match,
        "hallucination": hallu,
        "clarification": clarif,
    }

    # Official RAGAS metrics (only when retrieved chunks are available)
    if contexts:
        ctx_texts = [c.get("content", "") for c in contexts if c.get("content")]
        ragas = evaluate_ragas(
            question=case["input"],
            answer=answer,
            contexts=ctx_texts,
            ground_truth=gt.get("ground_truth", ""),
        )
        # Only store non-None scores in metrics (NaN from RAGAS → None → excluded from avg)
        if ragas["faithfulness"]   is not None:
            metrics["faithfulness"]   = ragas["faithfulness"]
        if ragas["context_recall"] is not None:
            metrics["context_recall"] = ragas["context_recall"]
        detail["ragas"] = ragas

    return {
        "id":     case["id"],
        "input":  case["input"],
        "answer": answer[:400],
        "metrics": metrics,
        "detail":  detail,
    }


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_answer(results: list[dict]) -> dict:
    if not results:
        return {}

    def avg(key: str) -> float:
        vals = [r["metrics"][key] for r in results]
        return round(sum(vals) / len(vals), 3)

    kf_coverages = [
        r["detail"]["match"]["key_facts"]["coverage"]
        for r in results
        if r.get("detail", {}).get("match", {}).get("key_facts", {}).get("coverage") is not None
    ]
    kf_coverage_avg = round(sum(kf_coverages) / len(kf_coverages), 3) if kf_coverages else None

    # RAGAS metrics — only average over cases where not skipped
    faith_scores  = [r["metrics"]["faithfulness"]   for r in results
                     if r.get("metrics", {}).get("faithfulness")   is not None]
    recall_scores = [r["metrics"]["context_recall"] for r in results
                     if r.get("metrics", {}).get("context_recall") is not None]

    out = {
        "n":                      len(results),
        "match_avg":              avg("match"),
        "match_full":             round(sum(1 for r in results if r["metrics"]["match"] == 2) / len(results), 3),
        "hallucination_clean":    avg("hallucination"),
        "clarification_acc":      avg("clarification"),
        "key_facts_coverage_avg": kf_coverage_avg,
    }
    if faith_scores:
        out["faithfulness_avg"]    = round(sum(faith_scores) / len(faith_scores), 3)
        out["context_recall_avg"]  = round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else None
        out["n_ragas"]             = len(faith_scores)
    return out
