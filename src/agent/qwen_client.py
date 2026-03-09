"""Shared Qwen client (DashScope / OpenAI-compatible)."""
from openai import OpenAI
import config

_client: OpenAI | None = None


def get_qwen_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL or None,
        )
    return _client
