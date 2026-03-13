"""Schema validation and deduplication for synthesized cases."""
from __future__ import annotations

_COMMON_REQUIRED = {"id", "layer_targets", "category", "dimensions", "context", "input"}

_LAYER_GT_FIELD = {
    "rewriter":  "rewriter_gt",
    "router":    "router_gt",
    "retrieval": "retrieval_gt",
    "answer":    "answer_gt",
}

_GT_REQUIRED_FIELDS: dict[str, set] = {
    "rewriter_gt": {
        "coref_map", "ellipsis_slots", "reference_rewrite",
        "required_entities", "forbidden_entities",
        "should_clarify", "is_prompt_injection",
    },
    "router_gt": {
        "expected_tools", "forbidden_tools", "no_tool_needed",
        "tool_params", "min_calls", "max_calls", "must_be_parallel",
    },
    "retrieval_gt": {"relevant_chunk_ids", "eval_at_k"},
    "answer_gt": {
        "ground_truth", "key_facts", "forbidden_content",
        "should_clarify", "is_safe",
    },
}

_DIMENSIONS_REQUIRED = {
    "is_multi_turn", "history_length", "has_coref",
    "depends_on_memory", "is_ambiguous", "should_clarify",
}

_CONTEXT_REQUIRED = {"user_profile", "memory_facts", "conversation_history"}


def validate_case(case: dict, layer: str) -> list[str]:
    errors: list[str] = []

    for f in _COMMON_REQUIRED:
        if f not in case:
            errors.append(f"Missing top-level field: {f}")

    if not isinstance(case.get("input"), str) or not case["input"].strip():
        errors.append("'input' must be a non-empty string")

    for f in _DIMENSIONS_REQUIRED:
        if f not in case.get("dimensions", {}):
            errors.append(f"dimensions.{f} missing")

    for f in _CONTEXT_REQUIRED:
        if f not in case.get("context", {}):
            errors.append(f"context.{f} missing")

    # GT block check
    if layer in _LAYER_GT_FIELD:
        gt_field = _LAYER_GT_FIELD[layer]
        gt = case.get(gt_field)
        if gt is None:
            errors.append(f"Missing GT block: {gt_field}")
        else:
            for f in _GT_REQUIRED_FIELDS[gt_field]:
                if f not in gt:
                    errors.append(f"{gt_field}.{f} missing")

    if not errors:
        errors.extend(_consistency_checks(case, layer))

    return errors


def _consistency_checks(case: dict, layer: str) -> list[str]:
    errors: list[str] = []
    dims = case.get("dimensions", {})

    # is_multi_turn requires history
    if dims.get("is_multi_turn") and dims.get("history_length", 0) == 0:
        errors.append("is_multi_turn=True but history_length=0")

    # Router consistency
    if layer == "router":
        rgt = case.get("router_gt", {})
        min_c, max_c = rgt.get("min_calls", 1), rgt.get("max_calls", 1)
        if min_c > max_c:
            errors.append(f"router_gt.min_calls ({min_c}) > max_calls ({max_c})")
        if rgt.get("must_be_parallel") and max_c < 2:
            errors.append("must_be_parallel=True but max_calls < 2")
        if rgt.get("no_tool_needed") and rgt.get("expected_tools"):
            errors.append("no_tool_needed=True but expected_tools is non-empty")

    return errors


# ─────────────────────────────────────────────────────────────
# Deduplication — character 4-gram Jaccard
# ─────────────────────────────────────────────────────────────

def _ngrams(text: str, n: int = 4) -> set[str]:
    t = text.lower()
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def is_duplicate(new_case: dict, existing: list[dict], threshold: float = 0.75) -> bool:
    ng = _ngrams(new_case.get("input", ""))
    for c in existing:
        inter = ng & _ngrams(c.get("input", ""))
        union = ng | _ngrams(c.get("input", ""))
        if union and len(inter) / len(union) >= threshold:
            return True
    return False
