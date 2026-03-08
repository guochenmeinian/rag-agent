import config
from prompts import rewriter as prompt
from .qwen_client import get_qwen_client


class QueryRewriter:
    """Rewrites user input into a standalone question using a small model."""

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

    def rewrite(self, user_input: str, context_prompt: str) -> str:
        user_content = (
            f"{context_prompt}\n\n"
            f"[用户最新输入]\n{user_input}\n\n"
            f"请改写成独立完整的问题："
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt.SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=300,
                extra_body={"enable_thinking": False},
            )
            refined = resp.choices[0].message.content.strip()
            return refined if len(refined) > 5 else user_input
        except Exception:
            return user_input
