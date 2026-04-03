import json
import logging
import os
import uuid
from typing import Generator

log = logging.getLogger(__name__)

import config
from .state import AgentState
from .memory import ConversationMemory
from .rewriter import QueryRewriter
from .planner import QueryPlanner
from .executor import AgentExecutor
from tools.registry import ToolRegistry



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
        self.planner = QueryPlanner(**(qwen_cfg or {}))
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
            {"type": "clarify",      "message": str}                            # rewriter short-circuit
            {"type": "refined",      "query": str}
            {"type": "tool_calling", "calls": [...], "source": "planner"|None}  # planner pre-fetch or executor
            {"type": "tool_done",    "results": list[dict], "source": ...}
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

        # --- Planner: decompose multi-topic queries before Executor loop ---
        if "planner" not in self.disabled:
            plan = self.planner.plan(refined_query, context_prompt)
            if plan.type == "decomposed" and plan.calls:
                # Build OpenAI-compatible tool_calls + registry calls
                fake_tool_calls = []
                registry_calls = []
                for spec in plan.calls:
                    call_id = f"plan_{uuid.uuid4().hex[:8]}"
                    # grep_search uses "keywords" field; all others use "query"
                    query_key = "keywords" if spec.tool == "grep_search" else "query"
                    args: dict = {query_key: spec.query}
                    if spec.car_model:
                        args["car_model"] = spec.car_model
                    fake_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": spec.tool,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })
                    registry_calls.append({"id": call_id, "name": spec.tool, "input": args})

                yield {
                    "type": "tool_calling",
                    "calls": [{"name": c["name"], **c["input"]} for c in registry_calls],
                    "source": "planner",
                }

                # Inject fake assistant tool_call message so tool results are valid
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": fake_tool_calls,
                })

                # Execute all planned calls in parallel
                tool_results = self.registry.run_parallel(registry_calls)

                # If ALL results failed, skip
                # injection — let the Executor loop decide tools from scratch instead of
                # reasoning over empty / error-only context.
                successful = [r for r in tool_results if r["result"].success]
                if not successful:
                    log.warning(
                        "planner: all %d pre-fetched calls failed, skipping injection",
                        len(tool_results),
                    )
                    # Roll back the fake assistant message we just appended
                    messages.pop()
                else:
                    state.tool_results.extend(tool_results)
                    yield {"type": "tool_done", "results": tool_results, "source": "planner"}

                    # Inject tool results into conversation
                    for r in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": r["id"],
                            "content": r["result"].to_llm_content(),
                        })

        while state.iteration < config.MAX_ITERATIONS:
            response = None
            for item_type, item_val in self.executor.run_stream(messages, extra_system=context_prompt):
                if item_type == "delta":
                    yield {"type": "text_delta", "text": item_val}
                else:
                    response = item_val
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
            state.answer = response.answer
            self.memory.add_message("assistant", state.answer)
            yield {
                "type": "done",
                "answer": state.answer,
                "tool_results": state.tool_results or None,
                "usage": dict(_usage),
            }
            self.memory.update_facts(user_input, state.answer)
            self._persist()
            return

        # Max iterations reached: force a direct answer if we only did tool calls
        if not state.answer:
            final = None
            for item_type, item_val in self.executor.run_stream(
                messages, extra_system=context_prompt, force_direct=True
            ):
                if item_type == "delta":
                    yield {"type": "text_delta", "text": item_val}
                else:
                    final = item_val
            state.answer = final.answer
            for k in _usage:
                _usage[k] += final.usage.get(k, 0)

        self.memory.add_message("assistant", state.answer)
        yield {
            "type": "done",
            "answer": state.answer,
            "tool_results": state.tool_results or None,
            "usage": dict(_usage),
        }
        self.memory.update_facts(user_input, state.answer)
        self._persist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def session_path(self) -> str | None:
        return self._session_path

    def _persist(self):
        if self._session_path:
            self.memory.save(self._session_path)
