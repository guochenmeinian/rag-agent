import json

from openai import OpenAI

import config
from prompts import executor as prompt
from .contracts import ExecutorResponse, ToolUseBlock


class AgentExecutor:
    def __init__(
        self,
        tool_schemas: list[dict],
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model or config.EXECUTOR_MODEL
        client_kwargs: dict = {"api_key": api_key or config.EXECUTOR_API_KEY}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._tool_schemas = tool_schemas

    def run(
        self,
        messages: list[dict],
        extra_system: str = "",
        force_direct: bool = False,
    ) -> ExecutorResponse:
        sys_content = f"{prompt.SYSTEM}\n\n{extra_system}" if extra_system else prompt.SYSTEM
        create_kwargs: dict = dict(
            model=self._model,
            max_tokens=4096,
            temperature=0.3,
            messages=[{"role": "system", "content": sys_content}] + messages,
        )
        if self._tool_schemas and not force_direct:
            create_kwargs["tools"] = self._tool_schemas
            create_kwargs["parallel_tool_calls"] = True
        elif self._tool_schemas and force_direct:
            create_kwargs["tools"] = self._tool_schemas
            create_kwargs["tool_choice"] = "none"
        # else: no tools at all → omit both, LLM answers directly
        response = self._client.chat.completions.create(**create_kwargs)
        return self._parse(response)

    def _parse(self, response) -> ExecutorResponse:
        msg = response.choices[0].message
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        if msg.tool_calls:
            blocks = [
                ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            raw = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            return ExecutorResponse(type="tool_call", raw_content=raw, tool_use_blocks=blocks, usage=usage)

        return ExecutorResponse(
            type="direct",
            answer=msg.content or "",
            raw_content={"role": "assistant", "content": msg.content},
            usage=usage,
        )
