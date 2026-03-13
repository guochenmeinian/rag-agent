import json
import logging
import config
from prompts import rewriter as prompt
from .contracts import RewriteResult
from .qwen_client import get_qwen_client

log = logging.getLogger(__name__)

class QueryRewriter:
    """Rewrites user input into a standalone question using a small model.

    Returns a dict with:
        type="rewrite"  → content is the rewritten query, pipeline continues normally
        type="clarify"  → content is a clarification message, pipeline short-circuits
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

    def rewrite(self, user_input: str, context_prompt: str) -> RewriteResult:
        user_content = (
            f"{context_prompt}\n\n"
            f"[用户最新输入]\n{user_input}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt.SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=300,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            raw = resp.choices[0].message.content.strip()
            return RewriteResult.parse(json.loads(raw), fallback=user_input)
        except Exception as e:
            log.warning("rewriter failed, falling back to raw input: %s", e)
        # Fallback: treat as plain rewrite
        return RewriteResult.rewrite(user_input)
