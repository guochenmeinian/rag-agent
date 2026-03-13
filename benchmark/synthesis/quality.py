"""Heuristic quality gates for synthesized benchmark cases."""
from __future__ import annotations

import re

_CAR_RE = re.compile(r"EC6|EC7|ES6|ES8|ET5T|ET5|ET7|ET9")
_PRONOUN_HINTS = ("它", "那款", "前者", "后者", "那个", "这台", "这辆")
_TECH_HINTS = ("续航", "快充", "轴距", "CLTC", "NOMI", "换电", "电池", "价格", "座", "悬架")
_INJECTION_HINTS = ("忽略", "扮演", "没有限制", "破解", "漏洞", "黑蔚来", "固件")


def check_case_quality(case: dict, target: dict) -> list[str]:
    errors: list[str] = []
    mode = target["mode"]
    dims = case.get("dimensions", {})
    context = case.get("context", {})
    history = context.get("conversation_history", [])
    user_input = (case.get("input") or "").strip()

    if len(user_input) < 8:
        errors.append("input too short to be a meaningful benchmark case")

    if mode == "single" and history:
        errors.append("single-mode case should not contain conversation history")
    if mode != "single" and not history:
        errors.append("multi-turn mode requires non-empty conversation history")

    if mode == "short":
        user_turns = sum(1 for msg in history if msg.get("role") == "user")
        if user_turns < 1 or user_turns > 4:
            errors.append("short-mode history length is outside expected range")

    if mode == "long_memory":
        user_turns = sum(1 for msg in history if msg.get("role") == "user")
        if user_turns < 4:
            errors.append("long-memory case does not have enough prior user turns")
        if not dims.get("depends_on_memory"):
            errors.append("long-memory case should set depends_on_memory=True")

    if dims.get("has_coref") and not any(hint in user_input for hint in _PRONOUN_HINTS):
        errors.append("coreference case lacks an obvious referring expression")

    if target["id"] == "rw_entity_extraction" and not any(hint in user_input for hint in _TECH_HINTS):
        errors.append("entity-extraction case lacks a recognizable technical topic")

    if target["id"] == "rw_clarify_safety" and not any(hint in user_input for hint in _INJECTION_HINTS):
        errors.append("safety case lacks a clear injection or harmful intent pattern")

    if dims.get("should_clarify") and target["id"] == "rw_clarify_ambiguous":
        car_mentions = _CAR_RE.findall(user_input)
        if len(set(car_mentions)) > 1:
            errors.append("ambiguous clarify case should not resolve itself by naming multiple cars in the test turn")

    return errors
