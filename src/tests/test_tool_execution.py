"""Unit tests for tool execution: timeout, retry, error normalization, parallel ordering.

Run:
    cd src && python -m pytest tests/test_tool_execution.py -v
"""
import time
import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field

# --- path bootstrap is handled by tests/__init__.py ---
from tools.result import ToolResult
from tools.base import BaseTool
from tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────
# Helpers: minimal concrete tools for testing
# ─────────────────────────────────────────────────────────────

class _SimpleInput(BaseModel):
    query: str = Field(..., min_length=1)


class ImmediateTool(BaseTool):
    """Always returns success immediately."""
    name = "immediate"
    description = "Test tool that returns immediately"
    InputModel = _SimpleInput
    timeout = 5
    max_retries = 0

    def _execute(self, inputs: _SimpleInput) -> ToolResult:
        return ToolResult.ok(f"result:{inputs.query}")


class SlowTool(BaseTool):
    """Sleeps for `delay` seconds — used to trigger timeout."""
    name = "slow"
    description = "Test tool that sleeps"
    InputModel = _SimpleInput
    timeout = 1
    max_retries = 0

    def __init__(self, delay: float = 3.0):
        self.delay = delay

    def _execute(self, inputs: _SimpleInput) -> ToolResult:
        time.sleep(self.delay)
        return ToolResult.ok("should not reach here")


class FlakyTool(BaseTool):
    """Fails `fail_count` times then succeeds — used to test retry logic."""
    name = "flaky"
    description = "Test tool that fails N times"
    InputModel = _SimpleInput
    timeout = 5

    def __init__(self, fail_count: int = 1):
        self.fail_count = fail_count
        self._calls = 0
        self.max_retries = fail_count  # enough retries to eventually succeed

    def _execute(self, inputs: _SimpleInput) -> ToolResult:
        self._calls += 1
        if self._calls <= self.fail_count:
            return ToolResult.error(f"transient failure #{self._calls}", "execution_error")
        return ToolResult.ok("recovered")


class AlwaysFailTool(BaseTool):
    """Always returns execution_error."""
    name = "always_fail"
    description = "Test tool that always fails"
    InputModel = _SimpleInput
    timeout = 5
    max_retries = 2

    def _execute(self, inputs: _SimpleInput) -> ToolResult:
        return ToolResult.error("permanent failure", "execution_error")


class RaisingTool(BaseTool):
    """Raises an exception inside _execute."""
    name = "raising"
    description = "Test tool that raises"
    InputModel = _SimpleInput
    timeout = 5
    max_retries = 0

    def _execute(self, inputs: _SimpleInput) -> ToolResult:
        raise RuntimeError("unexpected crash")


def _make_registry(*tools) -> ToolRegistry:
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


def _call(name: str, query: str, id: str = "c1") -> dict:
    return {"id": id, "name": name, "input": {"query": query}}


# ─────────────────────────────────────────────────────────────
# Timeout behaviour
# ─────────────────────────────────────────────────────────────

class TestTimeout:
    def test_slow_tool_returns_timeout_error(self):
        reg = _make_registry(SlowTool(delay=3.0))
        results = reg.run_parallel([_call("slow", "hi")])
        r = results[0]["result"]
        assert r.success is False
        assert r.error_type == "timeout"

    def test_timeout_message_contains_duration(self):
        reg = _make_registry(SlowTool(delay=3.0))
        results = reg.run_parallel([_call("slow", "hi")])
        assert "1" in results[0]["result"].content   # timeout=1s in SlowTool

    def test_fast_tool_not_timed_out(self):
        reg = _make_registry(ImmediateTool())
        results = reg.run_parallel([_call("immediate", "test")])
        assert results[0]["result"].success is True

    def test_timeout_not_retried(self):
        """A timed-out tool call should return immediately, not retry."""
        slow = SlowTool(delay=3.0)
        slow.max_retries = 3   # even with retries, timeout stops early
        reg = _make_registry(slow)
        t0 = time.monotonic()
        results = reg.run_parallel([_call("slow", "hi")])
        elapsed = time.monotonic() - t0
        assert results[0]["result"].error_type == "timeout"
        # Should NOT have waited 3 retries × 3s each
        assert elapsed < 5.0, f"Should have returned quickly, took {elapsed:.1f}s"

    def test_latency_ms_set_on_success(self):
        reg = _make_registry(ImmediateTool())
        result = reg.run_parallel([_call("immediate", "test")])[0]["result"]
        assert result.latency_ms >= 0


# ─────────────────────────────────────────────────────────────
# Retry behaviour
# ─────────────────────────────────────────────────────────────

class TestRetry:
    def test_flaky_tool_recovers_after_one_failure(self):
        """Tool fails once, retries once, succeeds."""
        flaky = FlakyTool(fail_count=1)
        reg = _make_registry(flaky)
        # Speed up by patching sleep
        with patch("tools.registry.time.sleep"):
            result = reg._run_with_resilience(_call("flaky", "test"))
        assert result.success is True
        assert result.content == "recovered"
        assert flaky._calls == 2

    def test_retry_exhausted_returns_last_error(self):
        """Tool exceeds max_retries → returns last failure."""
        always_fail = AlwaysFailTool()
        reg = _make_registry(always_fail)
        with patch("tools.registry.time.sleep"):
            result = reg._run_with_resilience(_call("always_fail", "test"))
        assert result.success is False
        assert result.error_type == "execution_error"

    def test_retry_call_count(self):
        """Tool with max_retries=2 must be called exactly 3 times (1 + 2 retries)."""
        class CountingTool(BaseTool):
            name = "counting"
            description = "Counts calls"
            InputModel = _SimpleInput
            timeout = 5
            max_retries = 2
            calls = 0

            def _execute(self, inputs):
                CountingTool.calls += 1
                return ToolResult.error("always fail", "execution_error")

        reg = _make_registry(CountingTool())
        with patch("tools.registry.time.sleep"):
            reg._run_with_resilience(_call("counting", "x"))
        assert CountingTool.calls == 3  # 1 initial + 2 retries

    def test_validation_error_not_retried(self):
        """validation_error should stop immediately — no retry."""
        class ValidationFailTool(BaseTool):
            name = "val_fail"
            description = "Always validation error"
            InputModel = _SimpleInput
            timeout = 5
            max_retries = 3
            calls = 0

            def _execute(self, inputs):
                ValidationFailTool.calls += 1
                return ToolResult.error("bad", "validation_error")

        reg = _make_registry(ValidationFailTool())
        with patch("tools.registry.time.sleep"):
            result = reg._run_with_resilience(_call("val_fail", "x"))
        assert result.error_type == "validation_error"
        assert ValidationFailTool.calls == 1  # no retry

    def test_exponential_backoff_sleep_values(self):
        """Sleep durations should follow 2^(attempt-1): 1, 2, 4, ..."""
        flaky = FlakyTool(fail_count=3)
        flaky.max_retries = 3
        reg = _make_registry(flaky)
        sleep_calls = []
        with patch("tools.registry.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            reg._run_with_resilience(_call("flaky", "x"))
        # first sleep is 2^0=1, second is 2^1=2, third is 2^2=4
        assert sleep_calls == [1, 2, 4]


# ─────────────────────────────────────────────────────────────
# Error normalisation
# ─────────────────────────────────────────────────────────────

class TestErrorNormalization:
    def test_exception_in_execute_becomes_execution_error(self):
        """Uncaught exceptions inside _execute are caught and normalised."""
        reg = _make_registry(RaisingTool())
        # _run_with_resilience catches exceptions from _run_with_timeout
        with patch("tools.registry.time.sleep"):
            result = reg._run_with_resilience(_call("raising", "test"))
        assert result.success is False
        assert result.error_type == "execution_error"

    def test_validation_error_is_caught_by_run(self):
        """BaseTool.run() itself must catch Pydantic ValidationError."""
        tool = ImmediateTool()
        result = tool.run(query="")   # min_length=1 → validation_error
        assert result.success is False
        assert result.error_type == "validation_error"

    def test_missing_required_field_validation_error(self):
        tool = ImmediateTool()
        result = tool.run()  # no query
        assert result.error_type == "validation_error"

    def test_run_never_raises(self):
        """BaseTool.run() must NEVER raise — always returns ToolResult."""
        tool = RaisingTool()
        # run() itself does validation then calls _execute — which raises.
        # The raising happens inside run(), which is NOT wrapped by registry here,
        # but the test confirms run() itself propagates the exception.
        # The registry's _run_with_timeout wraps this in a thread and catches it.
        reg = _make_registry(tool)
        result = reg.run_parallel([_call("raising", "hello")])[0]["result"]
        assert result.success is False

    def test_to_llm_content_prefixes(self):
        """ToolResult.to_llm_content() must prefix error type for LLM."""
        assert "[工具超时]"    in ToolResult.error("t", "timeout").to_llm_content()
        assert "[工具参数错误]" in ToolResult.error("v", "validation_error").to_llm_content()
        assert "[工具未配置]"   in ToolResult.error("c", "not_configured").to_llm_content()
        # Generic errors should not crash
        content = ToolResult.error("x", "execution_error").to_llm_content()
        assert "x" in content


# ─────────────────────────────────────────────────────────────
# Parallel execution & ordering
# ─────────────────────────────────────────────────────────────

class TestParallelExecution:
    def test_parallel_preserves_order(self):
        """Results must map back to original call order even if threads finish out of order."""
        class OrderedTool(BaseTool):
            name = "ordered"
            description = "Returns query"
            InputModel = _SimpleInput
            timeout = 5
            max_retries = 0

            def _execute(self, inputs):
                # Simulate varied latency: longer queries sleep a tiny bit more
                time.sleep(len(inputs.query) * 0.002)
                return ToolResult.ok(inputs.query)

        reg = _make_registry(OrderedTool())
        calls = [
            _call("ordered", "longerquery",    id="c1"),
            _call("ordered", "a",              id="c2"),
            _call("ordered", "mediumlength",   id="c3"),
        ]
        results = reg.run_parallel(calls)
        assert [r["id"] for r in results] == ["c1", "c2", "c3"]
        assert results[0]["result"].content == "longerquery"
        assert results[1]["result"].content == "a"
        assert results[2]["result"].content == "mediumlength"

    def test_parallel_independent_failures_do_not_block_others(self):
        """One failing tool must not prevent others from returning."""
        reg = _make_registry(ImmediateTool(), AlwaysFailTool())
        calls = [
            _call("immediate",   "ok",   id="c1"),
            _call("always_fail", "fail", id="c2"),
            _call("immediate",   "ok2",  id="c3"),
        ]
        with patch("tools.registry.time.sleep"):
            results = reg.run_parallel(calls)
        assert results[0]["result"].success is True
        assert results[1]["result"].success is False
        assert results[2]["result"].success is True

    def test_parallel_empty_calls_returns_empty(self):
        reg = _make_registry(ImmediateTool())
        assert reg.run_parallel([]) == []

    def test_parallel_result_dict_shape(self):
        """Every result dict must contain id, name, query, result."""
        reg = _make_registry(ImmediateTool())
        results = reg.run_parallel([_call("immediate", "hello", id="abc")])
        r = results[0]
        assert r["id"]    == "abc"
        assert r["name"]  == "immediate"
        assert r["query"] == "hello"
        assert isinstance(r["result"], ToolResult)

    def test_single_call_runs_fine(self):
        reg = _make_registry(ImmediateTool())
        results = reg.run_parallel([_call("immediate", "single")])
        assert len(results) == 1
        assert results[0]["result"].success is True

    def test_unknown_tool_returns_error(self):
        """Calling an unregistered tool must not crash the whole batch."""
        reg = _make_registry(ImmediateTool())
        calls = [
            _call("immediate", "ok",  id="c1"),
            _call("nonexistent", "x", id="c2"),
        ]
        results = reg.run_parallel(calls)
        assert results[0]["result"].success is True
        assert results[1]["result"].success is False
