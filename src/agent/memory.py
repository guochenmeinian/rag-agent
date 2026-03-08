import json
import os
from collections import deque

import config
from prompts.memory import SUMMARY_SYSTEM, EMPTY_SUMMARY
from .qwen_client import get_qwen_client


class ConversationMemory:
    """三层对话记忆：

    Layer 1 — system_prompt + user_profile  (静态，启动时设定)
    Layer 2 — context_summary               (结构化滚动摘要，recent 满时更新)
    Layer 3 — recent_messages               (最近 MAX_RECENT 条原文，滑动窗口)

    摘要格式（4个 slot）：
        [关注车型]   用户关注或比较过的车型
        [用户需求]   预算、用途、偏好
        [已确认数据] RAG检索到的具体参数
        [对话脉络]   最近话题转折
    """

    def __init__(self, system_prompt: str = "", user_profile: str = ""):
        self.system_prompt = system_prompt
        self.user_profile = user_profile
        self.context_summary: str = EMPTY_SUMMARY
        self.recent_messages: deque[dict] = deque(maxlen=config.MAX_RECENT_MESSAGES)
        self.ui_messages: list[dict] = []   # full chat UI history for frontend restore
        self._client = get_qwen_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str):
        """Add a message. When window is full, evict oldest into structured summary."""
        if len(self.recent_messages) == config.MAX_RECENT_MESSAGES:
            self._roll_into_summary(self.recent_messages[0])
        self.recent_messages.append({"role": role, "content": content})

    def format_for_prompt(self) -> str:
        """Serialize all three layers into a context block for rewriter and executor."""
        parts = []

        # Layer 1: system/user context
        sp = self.system_prompt
        if self.user_profile:
            sp = f"{sp}\n[用户背景] {self.user_profile}" if sp else f"[用户背景] {self.user_profile}"
        if sp:
            parts.append(sp)

        # Layer 2: structured summary
        parts.append(f"[对话记忆]\n{self.context_summary}")

        # Layer 3: verbatim recent
        if self.recent_messages:
            msgs = "\n".join(f"  {m['role']}: {m['content']}" for m in self.recent_messages)
            parts.append(f"[最近 {len(self.recent_messages)} 条消息]\n{msgs}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Serialize memory to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "system_prompt": self.system_prompt,
            "user_profile": self.user_profile,
            "context_summary": self.context_summary,
            "recent_messages": list(self.recent_messages),
            "ui_messages": self.ui_messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConversationMemory":
        """Restore memory from a JSON file."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupted file — start fresh and remove the broken file
            try:
                os.remove(path)
            except OSError:
                pass
            return cls()
        mem = cls(
            system_prompt=data.get("system_prompt", ""),
            user_profile=data.get("user_profile", ""),
        )
        mem.context_summary = data.get("context_summary", EMPTY_SUMMARY)
        for msg in data.get("recent_messages", []):
            mem.recent_messages.append(msg)
        mem.ui_messages = data.get("ui_messages", [])
        return mem

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _roll_into_summary(self, evicted: dict):
        """Update structured summary by integrating the evicted message via Qwen."""
        new_line = f"{evicted['role']}: {evicted['content']}"
        user_content = (
            f"【当前摘要】\n{self.context_summary}\n\n"
            f"【新消息】\n{new_line}\n\n"
            f"请输出更新后的摘要（保持4个字段格式）："
        )

        try:
            resp = self._client.chat.completions.create(
                model=config.QWEN_MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=300,
                extra_body={"enable_thinking": False},
            )
            updated = resp.choices[0].message.content.strip()
            # Sanity check: must contain at least one slot header
            if "[关注车型]" in updated:
                self.context_summary = updated
            else:
                # Fallback: append raw text to 对话脉络 slot
                self.context_summary += f"\n(+) {new_line[:80]}"
        except Exception:
            self.context_summary += f"\n(+) {new_line[:80]}"
