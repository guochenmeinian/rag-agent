"""BaseTool — Pydantic-validated, schema-auto-generating base class."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from .result import ToolResult


# ─────────────────────────────────────────────────────────────
# Schema generation
# ─────────────────────────────────────────────────────────────

def _resolve_ref(schema: dict, defs: dict[str, dict]) -> dict:
    """Recursively resolve $ref pointers in a JSON schema fragment."""
    if "$ref" in schema:
        ref_key = schema["$ref"].split("/")[-1]
        resolved = dict(defs.get(ref_key, {}))
        # Merge non-$ref keys from the original (e.g. description on the field site)
        for k, v in schema.items():
            if k != "$ref":
                resolved.setdefault(k, v)
        return resolved
    return schema


def pydantic_to_openai_schema(
    name: str,
    description: str,
    model: type[BaseModel],
) -> dict:
    """Auto-generate an OpenAI function-calling schema from a Pydantic model.

    Supports: basic types, Literal (enum), Field(description=..., min_length=...).
    """
    full = model.model_json_schema()
    defs = full.get("$defs", {})

    props: dict[str, dict] = {}
    for field_name, raw in full.get("properties", {}).items():
        resolved = _resolve_ref(raw, defs)

        prop: dict[str, Any] = {}

        # Type
        if "type" in resolved:
            prop["type"] = resolved["type"]
        elif "enum" in resolved:
            prop["type"] = "string"
        else:
            prop["type"] = "string"

        # Semantics
        if "description" in resolved:
            prop["description"] = resolved["description"]
        if "enum" in resolved:
            prop["enum"] = resolved["enum"]

        # Constraints (informational for the LLM)
        for constraint in ("minLength", "maxLength", "minimum", "maximum", "pattern"):
            if constraint in resolved:
                prop[constraint] = resolved[constraint]

        props[field_name] = prop

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": full.get("required", []),
            },
        },
    }


# ─────────────────────────────────────────────────────────────
# BaseTool
# ─────────────────────────────────────────────────────────────

class BaseTool(ABC):
    """Abstract base for all tools.

    Subclasses must declare:
        name        str
        description str
        InputModel  type[BaseModel]   ← Pydantic model; drives validation + schema

    Optionally override:
        timeout     int   seconds before execution is considered failed (default 30)
        max_retries int   retry attempts on non-validation failures (default 1)
        cacheable   bool  hint for the registry to cache results (default False)
    """

    name: str
    description: str
    InputModel: type[BaseModel]

    timeout:     int  = 30
    max_retries: int  = 1
    cacheable:   bool = False

    # ── Public entry point ────────────────────────────────────

    def run(self, **kwargs) -> ToolResult:
        """Validate inputs → execute → stamp latency. Never raises."""
        t0 = time.monotonic()

        # 1. Validate
        try:
            validated = self.InputModel(**kwargs)
        except ValidationError as exc:
            first = exc.errors()[0]
            msg = f"字段 '{first['loc'][0]}': {first['msg']}"
            return ToolResult(
                content=f"参数验证失败 — {msg}",
                success=False,
                error_type="validation_error",
                latency_ms=0,
            )

        # 2. Execute
        result = self._execute(validated)
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        return result

    # ── Abstract ──────────────────────────────────────────────

    @abstractmethod
    def _execute(self, inputs: BaseModel) -> ToolResult:
        """Implement tool logic here. Inputs are guaranteed valid."""

    # ── Schema (auto-generated) ───────────────────────────────

    @property
    def schema(self) -> dict:
        """OpenAI function-calling schema, auto-derived from InputModel."""
        return pydantic_to_openai_schema(self.name, self.description, self.InputModel)
