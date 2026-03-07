import os
import json
import re
from .qwen_client import get_qwen_client

SMALL_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-instruct")

_SYSTEM = """你是一个回答质量评审员，判断回答是否基于提供的检索结果，而不是依赖先验知识。

判断标准：
1. 若提供了检索结果，回答中的具体数字/参数必须来自检索结果，不能自行编造
2. 若回答引用了检索结果中没有的具体数据，判定为不通过
3. 若未提供检索结果，检查回答是否合理承认了不确定性，避免过度自信

输出严格为 JSON 格式（不要有其他内容）：
{"pass": true, "feedback": "简要说明"}
或
{"pass": false, "feedback": "指出具体问题，告诉模型如何改进"}"""


class Reflector:
    """Evaluates answer grounding: checks if answer is based on retrieved results,
    not prior knowledge."""

    def __init__(self):
        self._client = get_qwen_client()

    def reflect(
        self,
        query: str,
        answer: str,
        tool_results: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """
        Returns:
            (passed, feedback) — if passed is False, feedback explains what to fix.
        """
        tool_context = ""
        if tool_results:
            retrieved = "\n---\n".join(r["result"] for r in tool_results)
            tool_context = f"\n\n[检索结果]\n{retrieved}"

        prompt = (
            f"[问题]\n{query}\n\n"
            f"[回答]\n{answer}"
            f"{tool_context}"
        )

        try:
            resp = self._client.chat.completions.create(
                model=SMALL_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            result = self._parse_json(raw)
            return bool(result.get("pass", True)), result.get("feedback", "")
        except Exception:
            return True, ""  # default pass on any error

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Try json.loads first; fall back to regex extraction."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Extract JSON object from surrounding text (e.g. markdown code blocks)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # Last resort: pattern match
        passed = "true" in text.lower() and "false" not in text.lower()
        return {"pass": passed, "feedback": text[:200]}
