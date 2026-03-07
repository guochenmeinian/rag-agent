import os
from collections import deque
from .qwen_client import get_qwen_client

SMALL_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-instruct")
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "6"))
MAX_RECENT = int(os.getenv("MAX_RECENT_MESSAGES", "5"))


class ConversationMemory:
    """Manages conversation context with proactive rolling summarization.

    Context structure (mirrors the flowchart):
        user_profile      - static user info
        context_summary   - compressed summary of older turns (updated every SUMMARY_INTERVAL msgs)
        recent_messages   - last MAX_RECENT raw messages (sliding window)
    """

    def __init__(self, user_profile: str = ""):
        self.user_profile = user_profile
        self.context_summary = ""
        self.recent_messages: deque[dict] = deque(maxlen=MAX_RECENT)
        self._pending: list[dict] = []  # buffer waiting to be summarized

        self._client = get_qwen_client()

    def add_message(self, role: str, content: str):
        msg = {"role": role, "content": content}
        self.recent_messages.append(msg)
        self._pending.append(msg)

        if len(self._pending) >= SUMMARY_INTERVAL:
            self._update_summary()

    def _update_summary(self):
        if not self._pending:
            return

        pending_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in self._pending
        )
        prev = f"已有摘要：\n{self.context_summary}\n\n" if self.context_summary else ""

        prompt = (
            f"{prev}请将以下新对话内容压缩为简洁摘要（100字以内），"
            f"保留关键信息和用户意图：\n\n{pending_text}"
        )

        resp = self._client.chat.completions.create(
            model=SMALL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        self.context_summary = resp.choices[0].message.content.strip()
        self._pending.clear()

    def format_for_prompt(self) -> str:
        """Format context as a prompt string for the rewriter and executor."""
        parts = []
        if self.user_profile:
            parts.append(f"[用户信息]\n{self.user_profile}")
        if self.context_summary:
            parts.append(f"[近期上下文总结]\n{self.context_summary}")
        if self.recent_messages:
            msgs = "\n".join(
                f"{m['role']}: {m['content']}" for m in self.recent_messages
            )
            parts.append(f"[最近 {len(self.recent_messages)} 条消息]\n{msgs}")
        return "\n\n".join(parts)
