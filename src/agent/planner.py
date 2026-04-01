import json
import logging

import config
from prompts import planner as prompt
from .contracts import PlanResult
from .qwen_client import get_qwen_client

log = logging.getLogger(__name__)


class QueryPlanner:
    """Analyses the refined query and decides whether to decompose into parallel sub-tasks.

    Returns a PlanResult:
        type="simple"      → let the Executor decide tools as usual
        type="decomposed"  → pre-execute the listed tool calls before Executor synthesis
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model or config.QWEN_MODEL
        if api_key or base_url:
            from openai import OpenAI
            client_kwargs: dict = {"api_key": api_key or config.QWEN_API_KEY}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
        else:
            self._client = get_qwen_client()

    def plan(self, refined_query: str, context_prompt: str) -> PlanResult:
        user_content = (
            f"{context_prompt}\n\n"
            f"[问题]\n{refined_query}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt.SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=500,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            raw = resp.choices[0].message.content.strip()
            return PlanResult.parse(json.loads(raw))
        except Exception as e:
            log.warning("planner failed, falling back to simple: %s", e)
        return PlanResult.simple()
