"""Shared Qwen client via DashScope (OpenAI-compatible)."""
import os
from openai import OpenAI

_client: OpenAI | None = None


def get_qwen_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _client
