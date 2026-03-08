from typing import Generator
from .state import AgentState
from .memory import ConversationMemory
from .planner import QueryRewriter
from .executor import AgentExecutor
from .reflector import Reflector
from tools.registry import ToolRegistry


class AgentWorkflow:
    """Main orchestrator.

    user_input
      → memory (context + proactive rolling summary)
      → Qwen rewriter  →  refined standalone question
      → Claude executor
            tool_call  →  parallel tool runs  →  back to Claude
            direct     →  Qwen reflector (grounding check)
                              pass  →  save to memory, yield done
                              fail  →  re-run Claude with feedback
    """

    def __init__(self, registry: ToolRegistry, user_profile: str = ""):
        self.memory = ConversationMemory(user_profile=user_profile)
        self.rewriter = QueryRewriter()
        self.executor = AgentExecutor(tool_schemas=registry.schemas)
        self.reflector = Reflector()
        self.registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Synchronous wrapper — consumes the stream and returns the final answer."""
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
        # 1. Update memory (may trigger proactive summary)
        self.memory.add_message("user", user_input)
        context_prompt = self.memory.format_for_prompt()

        # 2. Rewrite with Qwen
        yield {"type": "rewriting"}
        refined_query = self.rewriter.rewrite(user_input, context_prompt)
        yield {"type": "refined", "query": refined_query}

        state = AgentState(user_input=user_input, refined_query=refined_query)
        messages = [
            {"role": "user", "content": f"{context_prompt}\n\n[当前问题]\n{refined_query}"}
        ]

        while state.iteration < state.MAX_ITER:
            response = self.executor.run(messages)

            # --- Tool call branch ---
            if response.type == "tool_call":
                calls = [
                    {"id": b.id, "name": b.name, "input": b.input}
                    for b in response.tool_use_blocks
                ]
                # Emit human-readable call info for UI
                yield {
                    "type": "tool_calling",
                    "calls": [{"name": c["name"], **c["input"]} for c in calls],
                }

                messages.append(response.raw_content)
                tool_results = self.registry.run_parallel(calls)
                state.tool_results = tool_results
                yield {"type": "tool_done", "results": tool_results}

                for r in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": r["id"],
                        "content": r["result"],
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
                yield {
                    "type": "done",
                    "answer": state.answer,
                    "tool_results": state.tool_results or None,
                }
                return

            # Reflection failed: feed back to Claude
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

        # Max iterations: return best-effort answer
        self.memory.add_message("assistant", state.answer)
        yield {
            "type": "done",
            "answer": state.answer,
            "tool_results": state.tool_results or None,
        }
