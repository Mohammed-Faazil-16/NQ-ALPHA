def build_prompt(user_input: str, memory_context: list[str]) -> str:
    memory_block = "\n".join(memory_context) if memory_context else "None"
    return f"""
SYSTEM:
You are a quantitative financial AI assistant.

- Be direct
- Be analytical
- Avoid disclaimers
- Provide structured outputs

MEMORY:
{memory_block}

USER:
{user_input}
""".strip()
