import os
from dataclasses import dataclass, field
from typing import Literal
import anthropic

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

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
class ExecutorResponse:
    type: Literal["tool_call", "direct"]
    answer: str = ""
    # Raw content blocks from Anthropic response (needed for message reconstruction)
    raw_content: list = field(default_factory=list)
    # tool_use blocks only, subset of raw_content
    tool_use_blocks: list = field(default_factory=list)


class AgentExecutor:
    def __init__(self, tool_schemas: list[dict]):
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._tool_schemas = tool_schemas

    def run(self, messages: list[dict]) -> ExecutorResponse:
        """Single Claude API call. messages is the full conversation for this turn."""
        response = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=_SYSTEM,
            tools=self._tool_schemas,
            messages=messages,
        )
        return self._parse(response)

    def _parse(self, response) -> ExecutorResponse:
        tool_use_blocks = []
        text_parts = []

        for block in response.content:
            if block.type == "tool_use":
                tool_use_blocks.append(block)
            elif block.type == "text":
                text_parts.append(block.text)

        if tool_use_blocks:
            return ExecutorResponse(
                type="tool_call",
                raw_content=response.content,
                tool_use_blocks=tool_use_blocks,
            )

        return ExecutorResponse(
            type="direct",
            answer="\n".join(text_parts).strip(),
            raw_content=response.content,
        )
