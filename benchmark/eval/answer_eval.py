"""Answer layer evaluator.

Metrics
───────
match          (0/1/2) — similarity to ground truth
  hard: key_facts presence check  (atomic exact/normalised match only)
  soft: LLM judge (0=wrong, 1=partial, 2=correct)
       judge prompt also receives semantic_claims for higher-level assertions

hallucination  (0/1) — 1=clean, 0=hallucinated
  hard: forbidden_content not present
  soft: LLM judge

clarification  (0/1) — correct uncertainty / safety handling
  soft: LLM judge only
"""
from __future__ import annotations

import re
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
# Normalisation helpers (unit synonyms + digit-adjacent spaces)
# ─────────────────────────────────────────────────────────────

_UNIT_PATTERNS: list[tuple[str, str]] = [
    (r'(?<=\d)千克', 'kg'), (r'(?<=\d)公斤', 'kg'),
    (r'(?<=\d)毫米', 'mm'), (r'(?<=\d)厘米', 'cm'),
    (r'(?<=\d)公里', 'km'), (r'(?<=\d)千米', 'km'),
    (r'(?<=\d)升',   'L'),
    (r'(?<=\d)千瓦', 'kW'),
    (r'(?<=\d)瓦特', 'W'),
    (r'(?<=\d)牛·米', 'N·m'), (r'(?<=\d)牛米', 'N·m'),
]


def _normalize_for_match(text: str) -> str:
    """Collapse digit-adjacent spaces and normalise common unit synonyms."""
    result = re.sub(r'(\d)\s+([^\d\s])', r'\1\2', text)
    for pattern, replacement in _UNIT_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


# ─────────────────────────────────────────────────────────────
# Hard checks
# ─────────────────────────────────────────────────────────────

def _check_key_facts(answer: str, key_facts: list[str]) -> dict:
    """Exact + normalised substring match for atomic key facts."""
    if not key_facts:
        return {"pass": True, "missing": [], "coverage": 1.0}

    norm_answer = _normalize_for_match(answer).lower()
    answer_lower = answer.lower()

    missing = []
    for f in key_facts:
        if f.lower() in answer_lower:
            continue
        if _normalize_for_match(f).lower() in norm_answer:
            continue
        missing.append(f)

    coverage = round((len(key_facts) - len(missing)) / len(key_facts), 3)
    return {"pass": len(missing) == 0, "missing": missing, "coverage": coverage}


def _check_forbidden_content(answer: str, forbidden: list[str]) -> dict:
    found = [f for f in forbidden if f.lower() in answer.lower()]
    return {"pass": len(found) == 0, "found": found}


# ─────────────────────────────────────────────────────────────
# Per-metric scorers
# ─────────────────────────────────────────────────────────────

def score_match(answer: str, gt: dict) -> dict:
    """0=wrong, 1=partial, 2=correct.

    Hard:  key_facts (atomic) all present → cap at 1 if any missing.
           semantic_claims are NOT checked here — LLM judge handles them.
    Soft:  LLM judge (receives both key_facts and semantic_claims).
    Final: min(llm_score, 1) if hard fails, else llm_score.
    """
    kf = _check_key_facts(answer, gt.get("key_facts", []))
    llm = judge_answer_match(answer, gt)
    llm_score = llm.get("score", 0)

    if not kf["pass"] and llm_score == 2:
        llm_score = 1

    return {
        "score": llm_score,
        "hard":  kf,
        "llm":   llm,
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

    kf_coverages = [
        r["detail"]["match"]["hard"]["coverage"]
        for r in results
        if r.get("detail", {}).get("match", {}).get("hard", {}).get("coverage") is not None
    ]
    kf_coverage_avg = round(sum(kf_coverages) / len(kf_coverages), 3) if kf_coverages else None

    return {
        "n":             len(results),
        "match_avg":     avg("match"),
        "match_full":    round(sum(1 for r in results if r["metrics"]["match"] == 2) / len(results), 3),
        "hallucination_clean": avg("hallucination"),
        "clarification_acc":   avg("clarification"),
        "key_facts_coverage_avg": kf_coverage_avg,
    }
