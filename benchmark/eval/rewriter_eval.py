"""Rewriter layer evaluator.

Metrics
───────
standalone                (0/1)  应用于 should_clarify=False 的 case
  hard: coref_map 里的 surface 词消失、ellipsis_slots 被填充
  soft: LLM judge

entity_extraction_accuracy (0/1) 应用于 should_clarify=False 的 case
  hard: required_entities 全部出现在改写中
  soft: LLM judge

clarify_detection          (0/1) 应用于 should_clarify=True 的 case
  hard only: rewriter 输出 type=clarify → 1，输出 type=rewrite → 0
  （是/否的判断不需要 LLM judge）

对于不适用某指标的 case，该指标返回 None，聚合时分母只计非 None 的 case。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from benchmark.eval.llm_judge import judge_standalone, judge_entity_extraction


# ─────────────────────────────────────────────────────────────
# Hard (rule-based) checks
# ─────────────────────────────────────────────────────────────

_CAR_MODEL_RE = re.compile(r"EC[67]|ES[68]|ET5T?|ET[79]")


def _check_coref_resolved(rewrite: str, coref_map: dict) -> dict:
    failures = []
    for surface, resolved in coref_map.items():
        if surface.lower() in rewrite.lower():
            failures.append(f"'{surface}' still present (not resolved)")
        if resolved.lower() not in rewrite.lower():
            failures.append(f"'{resolved}' (resolved referent) missing")
    return {"pass": not failures, "failures": failures}


def _check_ellipsis_filled(rewrite: str, slots: list) -> dict:
    if not slots:
        return {"pass": True, "failures": []}
    failures = []
    for slot in slots:
        if slot == "车型" and not _CAR_MODEL_RE.search(rewrite):
            failures.append("slot '车型' not filled — no car model in rewrite")
    return {"pass": not failures, "failures": failures}


def _check_required_entities(rewrite: str, entities: list) -> dict:
    missing = [e for e in entities if e.lower() not in rewrite.lower()]
    return {"pass": not missing, "missing": missing}


def _check_forbidden_entities(rewrite: str, entities: list) -> dict:
    found = [e for e in entities if e.lower() in rewrite.lower()]
    return {"pass": not found, "found": found}


# ─────────────────────────────────────────────────────────────
# Per-metric scorers
# ─────────────────────────────────────────────────────────────

def score_standalone(rewrite: str, context: dict, gt: dict) -> dict:
    coref    = _check_coref_resolved(rewrite, gt.get("coref_map", {}))
    ellipsis = _check_ellipsis_filled(rewrite, gt.get("ellipsis_slots", []))
    hard_pass = coref["pass"] and ellipsis["pass"]

    llm = judge_standalone(rewrite, context, gt)
    score = int(hard_pass and llm.get("score", 0) == 1)

    return {"score": score, "hard": {"pass": hard_pass, "coref": coref, "ellipsis": ellipsis}, "llm": llm}


def score_entity_extraction(rewrite: str, gt: dict) -> dict:
    required  = _check_required_entities(rewrite, gt.get("required_entities", []))
    forbidden = _check_forbidden_entities(rewrite, gt.get("forbidden_entities", []))
    hard_pass = required["pass"] and forbidden["pass"]

    llm = judge_entity_extraction(rewrite, gt)
    score = int(hard_pass and llm.get("score", 0) == 1)

    return {"score": score, "hard": {"pass": hard_pass, "required": required, "forbidden": forbidden}, "llm": llm}


# ─────────────────────────────────────────────────────────────
# Case-level runner
# ─────────────────────────────────────────────────────────────

def eval_rewriter_case(case: dict, rewrite_result: dict) -> dict:
    """Evaluate one rewriter case.

    Args:
        case:           BenchmarkCase with rewriter_gt and context
        rewrite_result: dict from QueryRewriter.rewrite()
                        {"type": "rewrite", "content": "..."}
                     or {"type": "clarify", "content": "..."}

    Returns:
        metrics dict where inapplicable metrics are None.
    """
    gt           = case.get("rewriter_gt", {})
    context      = case.get("context", {})
    should_clarify = gt.get("should_clarify", False)

    output_type    = rewrite_result.get("type", "rewrite")
    output_content = rewrite_result.get("content", "")

    metrics: dict = {
        "standalone":                 None,
        "entity_extraction_accuracy": None,
        "clarify_detection":          None,
    }
    detail: dict = {}

    if should_clarify:
        # Only clarify_detection applies — pure hard check
        correct = int(output_type == "clarify")
        metrics["clarify_detection"] = correct
        detail["clarify_detection"] = {
            "score":    correct,
            "expected": "clarify",
            "got":      output_type,
            "content":  output_content,
        }

    else:
        # standalone + entity_extraction apply
        if output_type == "clarify":
            # rewriter incorrectly chose to clarify → both fail immediately
            metrics["standalone"] = 0
            metrics["entity_extraction_accuracy"] = 0
            detail["standalone"] = {"score": 0, "note": "rewriter incorrectly output clarify"}
            detail["entity_extraction_accuracy"] = {"score": 0, "note": "rewriter incorrectly output clarify"}
        else:
            sa = score_standalone(output_content, context, gt)
            en = score_entity_extraction(output_content, gt)
            metrics["standalone"]                 = sa["score"]
            metrics["entity_extraction_accuracy"] = en["score"]
            detail["standalone"]                  = sa
            detail["entity_extraction_accuracy"]  = en

    return {
        "id":            case["id"],
        "category":      case.get("category", ""),
        "input":         case["input"],
        "rewrite_result": rewrite_result,
        "metrics":       metrics,
        "detail":        detail,
    }


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_rewriter(results: list[dict]) -> dict:
    def _avg(key: str) -> tuple[float, int]:
        vals = [r["metrics"][key] for r in results if r["metrics"].get(key) is not None]
        return (round(sum(vals) / len(vals), 3) if vals else 0.0, len(vals))

    sa_avg,  sa_n  = _avg("standalone")
    en_avg,  en_n  = _avg("entity_extraction_accuracy")
    cl_avg,  cl_n  = _avg("clarify_detection")

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, []).append(r)

    cat_summary = {}
    for cat, rows in by_category.items():
        cat_summary[cat] = {
            m: round(sum(r["metrics"][m] for r in rows if r["metrics"].get(m) is not None) /
                     max(1, sum(1 for r in rows if r["metrics"].get(m) is not None)), 3)
            for m in ("standalone", "entity_extraction_accuracy", "clarify_detection")
        }

    # Diagnostics: coref and ellipsis hard-check pass rates
    # Only over cases where standalone was evaluated (should_clarify=False)
    standalone_details = [
        r["detail"]["standalone"]["hard"]
        for r in results
        if r.get("detail", {}).get("standalone", {}).get("hard") is not None
    ]
    coref_rate = round(
        sum(1 for d in standalone_details if d["coref"]["pass"]) / len(standalone_details), 3
    ) if standalone_details else None
    ellipsis_rate = round(
        sum(1 for d in standalone_details if d["ellipsis"]["pass"]) / len(standalone_details), 3
    ) if standalone_details else None

    return {
        "n":                          len(results),
        "standalone":                 sa_avg,  "standalone_n":  sa_n,
        "entity_extraction_accuracy": en_avg,  "entity_n":      en_n,
        "clarify_detection":          cl_avg,  "clarify_n":     cl_n,
        # Diagnostics
        "coref_resolution_rate":  coref_rate,
        "ellipsis_fill_rate":     ellipsis_rate,
        "by_category":                cat_summary,
    }
