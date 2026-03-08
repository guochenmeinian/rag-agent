import os
import json
from dataclasses import dataclass, field
from typing import Literal
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_SYSTEM = """你是一个蔚来汽车智能助手，帮助用户了解蔚来车型信息和汽车选购决策。

你有两个工具：
- rag_search: 从蔚来专属知识库检索车型参数/规格/配置，必须指定 car_model（如EC6、ET5、ES8、ET7）
- web_search: 搜索实时网络信息，用于竞品对比、品牌资讯、非蔚来专属内容

工具使用原则：
1. 涉及蔚来具体车型参数（电池、续航、尺寸、配置等）→ rag_search，每个车型单独一个 call
2. 需要实时信息或非蔚来专属内容 → web_search
3. 同一问题涉及多个车型 → 并行发起多个 rag_search call
4. 纯知识性问题无需查询 → 直接回答

回答要求：基于检索结果给出有数据支撑的回答，不要凭先验知识猜测具体参数。"""


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
    def __init__(self, tool_schemas: list[dict]):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._tool_schemas = tool_schemas

    def run(self, messages: list[dict]) -> ExecutorResponse:
        response = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=4096,
            tools=self._tool_schemas,
            messages=[{"role": "system", "content": _SYSTEM}] + messages,
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
            return ExecutorResponse(
                type="tool_call",
                raw_content=raw,
                tool_use_blocks=blocks,
            )

        return ExecutorResponse(
            type="direct",
            answer=msg.content or "",
            raw_content={"role": "assistant", "content": msg.content},
        )
