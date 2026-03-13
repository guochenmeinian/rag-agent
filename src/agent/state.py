from dataclasses import dataclass, field


@dataclass
class AgentState:
    user_input: str
    refined_query: str = ""

    tool_results: list[dict] = field(default_factory=list)
    answer: str = ""

    iteration: int = 0
