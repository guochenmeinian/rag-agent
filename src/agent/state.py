from dataclasses import dataclass, field


@dataclass
class AgentState:
    user_input: str
    refined_query: str = ""

    tool_results: list[dict] = field(default_factory=list)
    answer: str = ""

    iteration: int = 0
    grep_rag_fallback_done: bool = False  # prevent double-retry for grep→rag fallback
    tool_call_batches: int = 0
    tool_call_count: int = 0
    force_direct_used: bool = False
