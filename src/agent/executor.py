import json
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI

import config
from prompts import executor as prompt


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ExecutorResponse:
    type: Literal["tool_call", "direct"]
    answer: str = ""
    raw_content: dict = field(default_factory=dict)
    tool_use_blocks: list = field(default_factory=list)


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

    def run(self, messages: list[dict]) -> ExecutorResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=4096,
            tools=self._tool_schemas,
            parallel_tool_calls=True,
            messages=[{"role": "system", "content": prompt.SYSTEM}] + messages,
        )
        return self._parse(response)

    def _parse(self, response) -> ExecutorResponse:
        msg = response.choices[0].message

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
            return ExecutorResponse(type="tool_call", raw_content=raw, tool_use_blocks=blocks)

        return ExecutorResponse(
            type="direct",
            answer=msg.content or "",
            raw_content={"role": "assistant", "content": msg.content},
        )
