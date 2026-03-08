from dataclasses import dataclass, field
import config


@dataclass
class AgentState:
    user_input: str
    refined_query: str = ""

    tool_results: list[dict] = field(default_factory=list)
    answer: str = ""
    reflection_feedback: str = ""

    iteration: int = 0
    MAX_ITER: int = field(default_factory=lambda: config.MAX_ITERATIONS)
