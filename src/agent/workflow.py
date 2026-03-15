import os
from typing import Generator

import config
from .state import AgentState
from .memory import ConversationMemory
from .rewriter import QueryRewriter
from .executor import AgentExecutor
from tools.registry import ToolRegistry


_NO_DATA_PATTERNS = ("未找到", "未能找到", "未能获取", "没有找到", "未获取到", "没有具体")


def _grep_hit_models(tool_results: list[dict]) -> list[str]:
    """Return car_models where grep_search returned content (result_count > 0)."""
    models: list[str] = []
    for tr in tool_results:
        if tr.get("name") != "grep_search":
            continue
        result = tr.get("result")
        meta = result.metadata if result is not None else {}
        if meta.get("result_count", 0) > 0:
            model = meta.get("car_model", "")
            if model and model not in models:
                models.append(model)
    return models


def _answer_indicates_no_data(answer: str) -> bool:
    return any(p in answer for p in _NO_DATA_PATTERNS)


class AgentWorkflow:
    """Main orchestrator.

    user_input
      → memory (structured rolling summary + recent messages)
      → Qwen rewriter  →  refined standalone question
                           clarify  →  short-circuit, return directly
      → GPT-4o executor
            tool_call  →  parallel tool runs  →  back to GPT-4o
            direct     →  save to memory, yield done
    """

    def __init__(
        self,
        registry: ToolRegistry,
        user_profile: str = "",
        system_prompt: str = "",
        session_id: str | None = None,
        executor_cfg: dict | None = None,
        qwen_cfg: dict | None = None,
        disabled: set[str] | None = None,
    ):
        """
        executor_cfg / qwen_cfg — optional runtime model overrides, e.g.:
            {"model": "moonshot-v1-8k", "api_key": "sk-...", "base_url": "https://api.moonshot.cn/v1"}
        """
        self.session_id = session_id
        self._session_path = (
            os.path.join(config.MEMORY_DIR, f"{session_id}.json")
            if session_id else None
        )

        # Restore memory from disk if session exists, else start fresh
        if self._session_path and os.path.exists(self._session_path):
            self.memory = ConversationMemory.load(self._session_path)
            # Allow caller to override profile/system_prompt on resume
            if user_profile:
                self.memory.user_profile = user_profile
                self.memory.global_user_info.raw = user_profile
            if system_prompt:
                self.memory.system_prompt = system_prompt
        else:
            self.memory = ConversationMemory(
                system_prompt=system_prompt,
                user_profile=user_profile,
            )

        self.rewriter = QueryRewriter(**(qwen_cfg or {}))
        self.executor = AgentExecutor(tool_schemas=registry.schemas, **(executor_cfg or {}))
        self.registry = registry
        self.disabled: set[str] = disabled or set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Synchronous wrapper — returns the final answer."""
        answer = ""
        for event in self.run_stream(user_input):
            if event["type"] == "done":
                answer = event["answer"]
        return answer

    def run_stream(self, user_input: str) -> Generator[dict, None, None]:
        """Generator that yields progress events for UI consumption.

        Event schema:
            {"type": "rewriting"}
            {"type": "clarify",      "message": str}          # rewriter short-circuit
            {"type": "refined",      "query": str}
            {"type": "tool_calling", "calls": [{"name": str, "query": str, ...}]}
            {"type": "tool_done",    "results": list[dict]}
            {"type": "done",         "answer": str, "tool_results": list[dict] | None}
        """
        self.memory.add_message("user", user_input)
        context_prompt = self.memory.format_for_prompt()

        if "rewriter" in self.disabled:
            refined_query = user_input
            yield {"type": "refined", "query": refined_query, "rewriter_skipped": True}
        else:
            yield {"type": "rewriting"}
            rewrite_result = self.rewriter.rewrite(user_input, context_prompt)

            # Short-circuit: rewriter decided to clarify/reject
            if rewrite_result.type == "clarify":
                msg = rewrite_result.content
                self.memory.add_message("assistant", msg)
                self._persist()
                yield {"type": "clarify", "message": msg}
                yield {"type": "done", "answer": msg, "tool_results": None}
                return

            refined_query = rewrite_result.content
            yield {"type": "refined", "query": refined_query}

        state = AgentState(user_input=user_input, refined_query=refined_query)
        messages = [{"role": "user", "content": refined_query}]
        _usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while state.iteration < config.MAX_ITERATIONS:
            response = self.executor.run(messages, extra_system=context_prompt)
            for k in _usage:
                _usage[k] += response.usage.get(k, 0)

            # --- Tool call branch ---
            if response.type == "tool_call":
                calls = [
                    {"id": b.id, "name": b.name, "input": b.input}
                    for b in response.tool_use_blocks
                ]
                yield {
                    "type": "tool_calling",
                    "calls": [{"name": c["name"], **c["input"]} for c in calls],
                }

                messages.append(response.raw_content)
                tool_results = self.registry.run_parallel(calls)
                state.tool_results.extend(tool_results)
                yield {"type": "tool_done", "results": tool_results}

                for r in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": r["id"],
                        "content": r["result"].to_llm_content(),
                    })
                state.iteration += 1
                continue

            # --- Direct answer branch ---
            # Option B fallback: grep returned content but executor says "not found"
            # → inject a rag_search retry instruction for the affected car models
            if (
                not state.grep_rag_fallback_done
                and _answer_indicates_no_data(response.answer)
            ):
                grep_models = _grep_hit_models(state.tool_results)
                if grep_models:
                    state.grep_rag_fallback_done = True
                    model_list = "、".join(grep_models)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[检索回退] grep_search 返回的内容不包含所需参数，"
                            f"请对以下车型改用 rag_search 重新检索：{model_list}。"
                            "请使用语义化的 query（如'续航里程'、'CLTC续航'、'快充功率'等），"
                            "不要再用电池规格词。"
                        ),
                    })
                    state.iteration += 1
                    continue

            state.answer = response.answer
            self.memory.add_message("assistant", state.answer)
            self.memory.update_facts(user_input, state.answer)
            self._persist()
            yield {
                "type": "done",
                "answer": state.answer,
                "tool_results": state.tool_results or None,
                "usage": dict(_usage),
            }
            return

        # Max iterations reached: force a direct answer if we only did tool calls
        if not state.answer:
            final = self.executor.run(messages, extra_system=context_prompt, force_direct=True)
            state.answer = final.answer
            for k in _usage:
                _usage[k] += final.usage.get(k, 0)

        self.memory.add_message("assistant", state.answer)
        self.memory.update_facts(user_input, state.answer)
        self._persist()
        yield {
            "type": "done",
            "answer": state.answer,
            "tool_results": state.tool_results or None,
            "usage": dict(_usage),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def session_path(self) -> str | None:
        return self._session_path

    def _persist(self):
        if self._session_path:
            self.memory.save(self._session_path)
