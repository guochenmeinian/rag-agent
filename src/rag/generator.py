import os
from typing import Optional, Any, List, Dict


def format_citations(items: List[Dict[str, Any]]) -> str:
    return "".join(f"[{x['rank']}] {x['text']}\n" for x in items)


def build_rag_prompt(query: str, formatted_references: str) -> str:
    return f"""
        你是一个智能助手，负责根据用户的问题和提供的参考内容生成回答。请严格按照以下要求生成回答：
        1. 回答必须基于提供的参考内容。
        2. 在回答中，每一块内容都必须标注引用来源，格式为：[引用编号]。
        3. 如果没有参考内容，请明确说明。若参考内容不足，可基于你自己的知识补充，但要清楚区分。

        参考内容：
        {formatted_references}

        用户问题：{query}
        """


def generate_answer(
    query: str,
    items: List[Dict[str, Any]],
    client=None,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.2,
) -> str:
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

    formatted_references = format_citations(items)
    prompt = build_rag_prompt(query=query, formatted_references=formatted_references)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    content: Optional[str] = completion.choices[0].message.content
    return content or ""

