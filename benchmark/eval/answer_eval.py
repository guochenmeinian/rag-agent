"""Answer layer evaluator.

Metrics
───────
match          (0/1/2) — similarity to ground truth
  hard: key_facts presence check
  soft: LLM judge (0=wrong, 1=partial, 2=correct)

hallucination  (0/1) — 1=clean, 0=hallucinated
  hard: forbidden_content not present
  soft: LLM judge

clarification  (0/1) — correct uncertainty / safety handling
  soft: LLM judge only
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from benchmark.eval.llm_judge import (
    judge_answer_match,
    judge_hallucination,
    judge_answer_clarification,
)


# ─────────────────────────────────────────────────────────────
# Hard checks
# ─────────────────────────────────────────────────────────────

def _check_key_facts(answer: str, key_facts: list[str]) -> dict:
    missing = [f for f in key_facts if f.lower() not in answer.lower()]
    return {"pass": len(missing) == 0, "missing": missing}


def _check_forbidden_content(answer: str, forbidden: list[str]) -> dict:
    found = [f for f in forbidden if f.lower() in answer.lower()]
    return {"pass": len(found) == 0, "found": found}


# ─────────────────────────────────────────────────────────────
# Per-metric scorers
# ─────────────────────────────────────────────────────────────

def score_match(answer: str, gt: dict) -> dict:
    """0=wrong, 1=partial, 2=correct.

    Hard: key_facts all present → hard floor of 1 if passed.
    Soft: LLM judge gives 0/1/2.
    Final = min(llm_score + hard_bonus, 2).
    """
    kf = _check_key_facts(answer, gt.get("key_facts", []))
    llm = judge_answer_match(answer, gt)
    llm_score = llm.get("score", 0)

    # If hard check fails (missing key facts), cap at 1
    if not kf["pass"] and llm_score == 2:
        llm_score = 1

    return {
        "score": llm_score,
        "hard": kf,
        "llm":  llm,
    }


def score_hallucination(answer: str, gt: dict) -> dict:
    """1 = clean, 0 = hallucinated.

    Hard: forbidden_content check. If fails → score=0 regardless of LLM.
    Soft: LLM judge.
    """
    fc = _check_forbidden_content(answer, gt.get("forbidden_content", []))
    llm = judge_hallucination(answer, gt)
    llm_pass = llm.get("score", 1) == 1

    final = int(fc["pass"] and llm_pass)

    return {
        "score": final,
        "hard": fc,
        "llm":  llm,
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

def eval_answer_case(case: dict, answer: str) -> dict:
    """Evaluate one answer case.

    Args:
        case:   BenchmarkCase dict with answer_gt
        answer: actual answer string from the system
    """
    gt = case.get("answer_gt", {})

    match  = score_match(answer, gt)
    hallu  = score_hallucination(answer, gt)
    clarif = score_clarification(answer, gt)

    return {
        "id":     case["id"],
        "input":  case["input"],
        "answer": answer[:400],
        "metrics": {
            "match":         match["score"],
            "hallucination": hallu["score"],
            "clarification": clarif["score"],
        },
        "detail": {
            "match":         match,
            "hallucination": hallu,
            "clarification": clarif,
        },
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

    return {
        "n":             len(results),
        "match_avg":     avg("match"),         # 0–2 scale
        "match_full":    round(sum(1 for r in results if r["metrics"]["match"] == 2) / len(results), 3),
        "hallucination_clean": avg("hallucination"),
        "clarification_acc":   avg("clarification"),
    }
