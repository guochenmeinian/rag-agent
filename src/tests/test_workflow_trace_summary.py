from agent.contracts import ExecutorResponse, RewriteResult, ToolUseBlock
from agent.workflow import AgentWorkflow
from tools.result import ToolResult


class _DummyRegistry:
    def __init__(self, results):
        self.schemas = []
        self._results = results

    def run_parallel(self, calls):
        return self._results


class _DummyRewriter:
    def __init__(self, *args, **kwargs):
        self._result = kwargs.pop("result", None)

    def rewrite(self, user_input: str, context_prompt: str) -> RewriteResult:
        return self._result or RewriteResult.rewrite(user_input)


class _DummyExecutor:
    def __init__(self, *args, **kwargs):
        self._responses = kwargs.pop("responses")

    def run(self, messages, extra_system: str = "", force_direct: bool = False):
        return self._responses.pop(0)


def test_run_stream_emits_trace_summary_for_tool_roundtrip(monkeypatch):
    tool_results = [
        {
            "id": "call_1",
            "name": "grep_search",
            "query": "ET5 续航",
            "result": ToolResult.ok("found", car_model="ET5", result_count=1),
        },
        {
            "id": "call_2",
            "name": "web_search",
            "query": "ET5 最新权益",
            "result": ToolResult.error("serper unavailable", "execution_error"),
        },
    ]

    responses = [
        ExecutorResponse(
            type="tool_call",
            tool_use_blocks=[
                ToolUseBlock(id="call_1", name="grep_search", input={"query": "ET5 续航"}),
                ToolUseBlock(id="call_2", name="web_search", input={"query": "ET5 最新权益"}),
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
        ExecutorResponse(
            type="direct",
            answer="ET5 的续航信息如下",
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        ),
    ]

    monkeypatch.setattr("agent.workflow.QueryRewriter", lambda **kwargs: _DummyRewriter(result=RewriteResult.rewrite("改写后的问题")))
    monkeypatch.setattr("agent.workflow.AgentExecutor", lambda **kwargs: _DummyExecutor(responses=list(responses)))

    workflow = AgentWorkflow(registry=_DummyRegistry(tool_results))
    monkeypatch.setattr(workflow.memory, "update_facts", lambda *args, **kwargs: None)

    events = list(workflow.run_stream("原始问题"))

    tool_calling = next(ev for ev in events if ev["type"] == "tool_calling")
    assert tool_calling["batch_size"] == 2
    assert tool_calling["tool_names"] == ["grep_search", "web_search"]
    assert tool_calling["iteration"] == 0

    tool_done = next(ev for ev in events if ev["type"] == "tool_done")
    assert tool_done["summary"]["success_count"] == 1
    assert tool_done["summary"]["error_count"] == 1
    assert tool_done["summary"]["error_types"] == {"execution_error": 1}
    assert tool_done["summary"]["tool_names"] == ["grep_search", "web_search"]

    done = next(ev for ev in events if ev["type"] == "done")
    assert done["usage"] == {"prompt_tokens": 17, "completion_tokens": 8, "total_tokens": 25}
    assert done["trace_summary"] == {
        "iterations": 1,
        "tool_call_batches": 1,
        "tool_call_count": 2,
        "tools_used": ["grep_search", "web_search"],
        "tool_success_count": 1,
        "tool_error_count": 1,
        "tool_error_types": {"execution_error": 1},
        "tool_latency_ms": 0,
        "grep_rag_fallback_used": False,
        "force_direct_used": False,
        "usage": {"prompt_tokens": 17, "completion_tokens": 8, "total_tokens": 25},
    }


def test_run_stream_emits_trace_summary_for_clarify(monkeypatch):
    monkeypatch.setattr("agent.workflow.QueryRewriter", lambda **kwargs: _DummyRewriter(result=RewriteResult.clarify("请先说明具体车型")))
    monkeypatch.setattr("agent.workflow.AgentExecutor", lambda **kwargs: _DummyExecutor(responses=[]))

    workflow = AgentWorkflow(registry=_DummyRegistry(results=[]))
    monkeypatch.setattr(workflow.memory, "update_facts", lambda *args, **kwargs: None)

    events = list(workflow.run_stream("帮我推荐"))

    done = next(ev for ev in events if ev["type"] == "done")
    assert done["answer"] == "请先说明具体车型"
    assert done["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert done["trace_summary"] == {
        "iterations": 0,
        "tool_call_batches": 0,
        "tool_call_count": 0,
        "tools_used": [],
        "tool_success_count": 0,
        "tool_error_count": 0,
        "tool_error_types": {},
        "tool_latency_ms": 0,
        "grep_rag_fallback_used": False,
        "force_direct_used": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
