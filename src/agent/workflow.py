import os
from typing import Generator

import config
from .state import AgentState
from .memory import ConversationMemory
from .planner import QueryRewriter
from .executor import AgentExecutor
from .reflector import Reflector
from tools.registry import ToolRegistry


class AgentWorkflow:
    """Main orchestrator.

    user_input
      → memory (structured rolling summary + recent messages)
      → Qwen rewriter  →  refined standalone question
      → GPT-4o executor
            tool_call  →  parallel tool runs  →  back to GPT-4o
            direct     →  Qwen reflector (grounding check)
                              pass  →  save to memory, yield done
                              fail  →  re-run with feedback
    """

    def __init__(
        self,
        registry: ToolRegistry,
        user_profile: str = "",
        system_prompt: str = "",
        session_id: str | None = None,
        executor_cfg: dict | None = None,
        qwen_cfg: dict | None = None,
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

        self.rewriter  = QueryRewriter(**(qwen_cfg or {}))
        self.executor  = AgentExecutor(tool_schemas=registry.schemas, **(executor_cfg or {}))
        self.reflector = Reflector(**(qwen_cfg or {}))
        self.registry = registry

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
            {"type": "refined",      "query": str}
            {"type": "tool_calling", "calls": [{"name": str, "query": str, ...}]}
            {"type": "tool_done",    "results": list[dict]}
            {"type": "reflecting"}
            {"type": "retry",        "feedback": str}
            {"type": "done",         "answer": str, "tool_results": list[dict] | None}
        """
        self.memory.add_message("user", user_input)
        context_prompt = self.memory.format_for_prompt()

        yield {"type": "rewriting"}
        refined_query = self.rewriter.rewrite(user_input, context_prompt)
        yield {"type": "refined", "query": refined_query}

        state = AgentState(user_input=user_input, refined_query=refined_query)
        messages = [
            {"role": "user", "content": f"{context_prompt}\n\n[当前问题]\n{refined_query}"}
        ]

        while state.iteration < config.MAX_ITERATIONS:
            response = self.executor.run(messages)

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
                state.tool_results.extend(tool_results)   # accumulate across rounds
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
            state.answer = response.answer
            yield {"type": "reflecting"}
            passed, feedback = self.reflector.reflect(
                query=state.refined_query,
                answer=state.answer,
                tool_results=state.tool_results or None,
            )

            if passed:
                self.memory.add_message("assistant", state.answer)
                self.memory.update_facts(user_input, state.answer)
                self._persist()
                yield {
                    "type": "done",
                    "answer": state.answer,
                    "tool_results": state.tool_results or None,
                }
                return

            # Reflection failed: feed back to model
            state.reflection_feedback = feedback
            yield {"type": "retry", "feedback": feedback}
            messages.append(response.raw_content)
            messages.append({
                "role": "user",
                "content": (
                    f"[质量反馈] {feedback}\n"
                    "请严格基于已检索到的信息重新回答，不要使用先验知识中的具体数字。"
                ),
            })
            state.iteration += 1

        # Max iterations reached: return best-effort answer
        self.memory.add_message("assistant", state.answer)
        self.memory.update_facts(user_input, state.answer)
        self._persist()
        yield {
            "type": "done",
            "answer": state.answer,
            "tool_results": state.tool_results or None,
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
