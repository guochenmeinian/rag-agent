"""Conversation memory: facts + global user info + recent messages."""

import json
import os
from collections import deque
from dataclasses import dataclass, field

import config
from prompts.memory import FACT_EXTRACT_SYSTEM
from .qwen_client import get_qwen_client


@dataclass
class GlobalUserInfo:
    """Global user profile — budget, family, preferences, focus models (optional)."""

    budget: str = ""
    family: str = ""
    preferences: str = ""
    focus_models: list[str] = field(default_factory=list)
    raw: str = ""  # backward compat with legacy user_profile string

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "family": self.family,
            "preferences": self.preferences,
            "focus_models": self.focus_models,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalUserInfo":
        info = cls()
        info.budget = d.get("budget", "")
        info.family = d.get("family", "")
        info.preferences = d.get("preferences", "")
        info.focus_models = d.get("focus_models") or []
        info.raw = d.get("raw", "")
        return info

    def summary(self) -> str:
        """One-line summary for prompts."""
        parts = []
        if self.budget:
            parts.append(f"预算{self.budget}")
        if self.family:
            parts.append(self.family)
        if self.preferences:
            parts.append(self.preferences)
        if self.focus_models:
            parts.append(f"关注{','.join(self.focus_models)}")
        if self.raw and not parts:
            return self.raw
        return "；".join(parts) if parts else ""


class ConversationMemory:
    """Conversation memory:

    - facts: atomic fact list (user-side, LLM extracted per turn)
    - global_user_info: optional structured profile (budget, family, etc.)
    - recent_messages: sliding window of raw messages
    """

    def __init__(self, system_prompt: str = "", user_profile: str = ""):
        self.system_prompt = system_prompt
        self.user_profile = user_profile
        self.global_user_info = GlobalUserInfo(raw=user_profile)
        self.facts: list[str] = []
        self.recent_messages: deque[dict] = deque(maxlen=config.MAX_RECENT_MESSAGES)
        self.ui_messages: list[dict] = []
        self._client = get_qwen_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str):
        """Add a message to the sliding window (no LLM call here)."""
        self.recent_messages.append({"role": role, "content": content})

    def update_facts(self, user_input: str, assistant_answer: str):
        """Extract/update facts from the just-completed turn. Called once per accepted answer."""
        current = "\n".join(self.facts) if self.facts else "无"
        user_content = (
            f"【当前事实列表】\n{current}\n\n"
            f"【本轮对话】\n"
            f"user: {user_input}\n"
            f"assistant: {assistant_answer[:800]}\n\n"
            f"请输出更新后的事实列表："
        )
        try:
            resp = self._client.chat.completions.create(
                model=config.QWEN_MODEL,
                messages=[
                    {"role": "system", "content": FACT_EXTRACT_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=200,
                extra_body={"enable_thinking": False},
            )
            raw = resp.choices[0].message.content.strip()
            if raw == "无" or not raw:
                self.facts = []
            else:
                self.facts = [line.strip() for line in raw.splitlines() if line.strip()]
        except Exception:
            pass  # keep existing facts on failure

    def format_for_prompt(self) -> str:
        """Serialize memory into context for rewriter and executor."""
        parts = []

        # [用户背景] — global_user_info or user_profile
        gui = self.global_user_info.summary()
        if gui:
            parts.append(f"[用户背景] {gui}")
        elif self.user_profile:
            parts.append(f"[用户背景] {self.user_profile}")

        # [用户记忆] — fact list
        if self.facts:
            bullet = "\n".join(f"- {f}" for f in self.facts)
            parts.append(f"[用户记忆]\n{bullet}")

        # [最近N条消息]
        if self.recent_messages:
            msgs = "\n".join(
                f"  {m['role']}: {m['content']}" for m in self.recent_messages
            )
            parts.append(f"[最近 {len(self.recent_messages)} 条消息]\n{msgs}")

        return "\n\n".join(parts)

    def get_memory_slots(self) -> dict[str, str]:
        """For UI display."""
        return {
            "用户背景": self.global_user_info.summary() or self.user_profile or "无",
            "事实列表": "; ".join(self.facts) if self.facts else "无",
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "system_prompt": self.system_prompt,
            "user_profile": self.user_profile,
            "global_user_info": self.global_user_info.to_dict(),
            "facts": self.facts,
            "recent_messages": list(self.recent_messages),
            "ui_messages": self.ui_messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConversationMemory":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            try:
                os.remove(path)
            except OSError:
                pass
            return cls()

        mem = cls(
            system_prompt=data.get("system_prompt", ""),
            user_profile=data.get("user_profile", ""),
        )

        if "global_user_info" in data:
            mem.global_user_info = GlobalUserInfo.from_dict(
                data["global_user_info"]
            )
        elif mem.user_profile:
            mem.global_user_info.raw = mem.user_profile

        if "facts" in data:
            mem.facts = data["facts"]
        elif "context_summary" in data:
            summary = data["context_summary"]
            lines = [
                l.strip()
                for l in summary.splitlines()
                if l.strip() and not l.startswith("[")
            ]
            mem.facts = lines

        for msg in data.get("recent_messages", []):
            mem.recent_messages.append(msg)
        mem.ui_messages = data.get("ui_messages", [])

        return mem
