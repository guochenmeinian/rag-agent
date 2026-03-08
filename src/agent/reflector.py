import json
import re

import config
from prompts import reflector as prompt
from .qwen_client import get_qwen_client


class Reflector:
    """Evaluates answer grounding: checks answer is based on retrieved results, not prior knowledge."""

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
            client_kwargs: dict = {"api_key": api_key or config.DASHSCOPE_API_KEY}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
        else:
            self._client = get_qwen_client()

    def reflect(
        self,
        query: str,
        answer: str,
        tool_results: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """Returns (passed, feedback). If passed is False, feedback explains what to fix."""
        tool_context = ""
        if tool_results:
            retrieved = "\n---\n".join(r["result"].to_llm_content() for r in tool_results)
            tool_context = f"\n\n[检索结果]\n{retrieved}"

        user_content = (
            f"[问题]\n{query}\n\n"
            f"[回答]\n{answer}"
            f"{tool_context}"
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt.SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            result = self._parse_json(raw)
            return bool(result.get("pass", True)), result.get("feedback", "")
        except Exception:
            return True, ""

    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        passed = "true" in text.lower() and "false" not in text.lower()
        return {"pass": passed, "feedback": text[:200]}
