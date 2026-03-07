from dataclasses import dataclass, field


@dataclass
class AgentState:
    user_input: str
    refined_query: str = ""

    # Tool execution
    tool_results: list[dict] = field(default_factory=list)  # [{id, name, query, result}]
    answer: str = ""

    # Reflection
    reflection_feedback: str = ""

    # Loop control
    iteration: int = 0
    MAX_ITER: int = 3
