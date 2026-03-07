def format_answer(answer: str, refined_query: str, tool_results: list[dict] | None) -> str:
    """Format the final answer for display.

    Optionally appends source references when tool results were used.
    """
    if not tool_results:
        return answer

    sources = []
    for r in tool_results:
        label = f"[{r['name']}] {r['query']}"
        if label not in sources:
            sources.append(label)

    source_line = " | ".join(sources)
    return f"{answer}\n\n---\n来源: {source_line}"
