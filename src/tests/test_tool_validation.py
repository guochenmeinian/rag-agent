"""Unit tests for tool input validation (Pydantic) and ToolResult structure.

Run:
    cd src && python -m pytest tests/test_tool_validation.py -v
"""
import pytest
from unittest.mock import patch

# --- path bootstrap is handled by tests/__init__.py ---
from tools.result import ToolResult
from tools.base import pydantic_to_openai_schema
from tools.rag_search import RagSearchTool, RagSearchInput
from tools.web_search import WebSearchTool, WebSearchInput


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def rag_tool():
    """RagSearchTool with no contexts (sufficient for validation tests)."""
    return RagSearchTool(contexts={})


@pytest.fixture
def web_tool():
    return WebSearchTool()


# ─────────────────────────────────────────────────────────────
# ToolResult structure
# ─────────────────────────────────────────────────────────────

class TestToolResult:
    def test_ok_factory(self):
        r = ToolResult.ok("some content", car_model="ET5")
        assert r.success is True
        assert r.content == "some content"
        assert r.metadata["car_model"] == "ET5"
        assert r.error_type is None

    def test_error_factory(self):
        r = ToolResult.error("something went wrong", "timeout")
        assert r.success is False
        assert r.error_type == "timeout"

    def test_to_llm_content_success(self):
        r = ToolResult.ok("normal result")
        assert r.to_llm_content() == "normal result"

    def test_to_llm_content_error_prefixed(self):
        r = ToolResult.error("请求超时", "timeout")
        llm = r.to_llm_content()
        assert "[工具超时]" in llm
        assert "请求超时" in llm

    def test_to_llm_content_validation_error(self):
        r = ToolResult.error("bad field", "validation_error")
        assert "[工具参数错误]" in r.to_llm_content()

    def test_to_llm_content_not_configured(self):
        r = ToolResult.error("need key", "not_configured")
        assert "[工具未配置]" in r.to_llm_content()

    def test_truncated_flag(self):
        r = ToolResult.ok("x" * 10, truncated=True)
        assert r.truncated is True

    def test_str_returns_content(self):
        r = ToolResult.ok("hello")
        assert str(r) == "hello"


# ─────────────────────────────────────────────────────────────
# RagSearchInput validation
# ─────────────────────────────────────────────────────────────

class TestRagSearchValidation:
    """Pydantic rejects invalid inputs BEFORE _execute is called."""

    VALID_PARAMS = {"query": "电池容量", "car_model": "ET5"}

    def test_valid_input_passes(self, rag_tool):
        # contexts is empty → not_found, but validation succeeds
        result = rag_tool.run(**self.VALID_PARAMS)
        assert result.error_type != "validation_error"

    @pytest.mark.parametrize("car_model", [
        "Tesla",        # not a NIO model
        "ET5 ET7",      # multiple models in one call
        "et5",          # wrong case
        "ET5/EC6",      # slash-separated
        "",             # empty
        "ET5 and EC6",  # natural language
    ])
    def test_invalid_car_model_rejected(self, rag_tool, car_model):
        result = rag_tool.run(query="续航", car_model=car_model)
        assert result.success is False
        assert result.error_type == "validation_error", (
            f"Expected validation_error for car_model={car_model!r}, got {result.error_type}"
        )

    def test_empty_query_rejected(self, rag_tool):
        result = rag_tool.run(query="", car_model="ET5")
        assert result.success is False
        assert result.error_type == "validation_error"

    def test_query_too_long_rejected(self, rag_tool):
        result = rag_tool.run(query="x" * 201, car_model="ET5")
        assert result.success is False
        assert result.error_type == "validation_error"

    def test_missing_car_model_rejected(self, rag_tool):
        result = rag_tool.run(query="续航")  # car_model missing
        assert result.success is False
        assert result.error_type == "validation_error"

    def test_missing_query_rejected(self, rag_tool):
        result = rag_tool.run(car_model="ET5")  # query missing
        assert result.success is False
        assert result.error_type == "validation_error"

    @pytest.mark.parametrize("car_model", [
        "EC6", "EC7", "ES6", "ES8", "ET5", "ET5T", "ET7", "ET9", "EL6"
    ])
    def test_all_valid_car_models_accepted(self, rag_tool, car_model):
        """Every model in the Literal set must pass validation."""
        result = rag_tool.run(query="续航", car_model=car_model)
        # validation_error = rejected; not_found = accepted (no context loaded)
        assert result.error_type != "validation_error", (
            f"car_model={car_model!r} should be valid but got validation_error"
        )


# ─────────────────────────────────────────────────────────────
# WebSearchInput validation
# ─────────────────────────────────────────────────────────────

class TestWebSearchValidation:
    def test_valid_query_passes(self, web_tool):
        # No API key → not_configured, but validation must pass
        with patch.object(type(web_tool), "timeout", 1):
            result = web_tool.run(query="特斯拉最新款")
        assert result.error_type != "validation_error"

    def test_empty_query_rejected(self, web_tool):
        result = web_tool.run(query="")
        assert result.success is False
        assert result.error_type == "validation_error"

    def test_query_too_long_rejected(self, web_tool):
        result = web_tool.run(query="x" * 301)
        assert result.success is False
        assert result.error_type == "validation_error"

    def test_missing_query_rejected(self, web_tool):
        result = web_tool.run()
        assert result.success is False
        assert result.error_type == "validation_error"


# ─────────────────────────────────────────────────────────────
# Schema auto-generation
# ─────────────────────────────────────────────────────────────

class TestSchemaGeneration:
    def test_rag_schema_structure(self, rag_tool):
        schema = rag_tool.schema
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "rag_search"
        params = fn["parameters"]
        assert "query" in params["properties"]
        assert "car_model" in params["properties"]
        assert set(params["required"]) == {"query", "car_model"}

    def test_rag_schema_car_model_has_enum(self, rag_tool):
        props = rag_tool.schema["function"]["parameters"]["properties"]
        car_prop = props["car_model"]
        assert "enum" in car_prop
        assert "ET5" in car_prop["enum"]
        assert "EC6" in car_prop["enum"]
        # Must NOT contain invalid models
        assert "Tesla" not in car_prop["enum"]

    def test_web_schema_structure(self, web_tool):
        schema = web_tool.schema
        fn = schema["function"]
        assert fn["name"] == "web_search"
        params = fn["parameters"]
        assert "query" in params["properties"]
        assert params["required"] == ["query"]

    def test_schema_query_has_description(self, rag_tool):
        props = rag_tool.schema["function"]["parameters"]["properties"]
        assert props["query"].get("description"), "query field should have a description"

    def test_schema_car_model_has_description(self, rag_tool):
        props = rag_tool.schema["function"]["parameters"]["properties"]
        assert props["car_model"].get("description"), "car_model field should have a description"

    def test_pydantic_to_openai_schema_roundtrip(self):
        from pydantic import BaseModel, Field
        class Dummy(BaseModel):
            text: str = Field(..., description="some text", min_length=2)
            num: int  = Field(..., description="a number")

        schema = pydantic_to_openai_schema("dummy", "A dummy tool", Dummy)
        props = schema["function"]["parameters"]["properties"]
        assert props["text"]["description"] == "some text"
        assert "num" in props
        assert schema["function"]["parameters"]["required"] == ["text", "num"]
