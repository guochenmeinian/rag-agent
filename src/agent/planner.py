import os
from .qwen_client import get_qwen_client

SMALL_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-instruct")

_SYSTEM = """你是一个对话理解助手。根据用户信息和对话历史，将用户最新输入改写成一个独立、完整、精准的问题。

要求：
- 补全省略的主语、宾语、车型等指代
- 将对话中的隐含意图融入问题
- 只输出改写后的问题，不要解释或多余内容

示例：
对话历史: user: 我想买车，在看蔚来  assistant: 好的，蔚来有多款车型
用户输入: EC6和ET5的电池容量差多少
改写输出: 我想买蔚来的车，纠结EC6和ET5，想知道这两款车的电池容量分别是多少，差距有多大。"""


class QueryRewriter:
    """Rewrites user input into a standalone question using Qwen (small model)."""

    def __init__(self):
        self._client = get_qwen_client()

    def rewrite(self, user_input: str, context_prompt: str) -> str:
        """
        Args:
            user_input: raw user message
            context_prompt: formatted context from ConversationMemory

        Returns:
            Standalone refined question, falls back to user_input on failure.
        """
        user_content = (
            f"{context_prompt}\n\n"
            f"[用户最新输入]\n{user_input}\n\n"
            f"请改写成独立完整的问题："
        )

        try:
            resp = self._client.chat.completions.create(
                model=SMALL_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=150,
            )
            refined = resp.choices[0].message.content.strip()
            return refined if len(refined) > 5 else user_input
        except Exception:
            return user_input
