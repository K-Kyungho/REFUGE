"""Thin wrapper around the Anthropic Messages API."""

from typing import Any, List, Tuple

SYSTEM_DS = "You are a data scientist for prediction task in relational database."
SYSTEM_JUDGE = "You are a rigorous judge for feature plan selection in relational databases."


def content_to_str(content) -> str:
    """Join an API response's content blocks (or a list of {'text': ...} dicts) into one string."""
    if not isinstance(content, list):
        return str(content)
    parts = []
    for b in content:
        if isinstance(b, dict) and "text" in b:
            parts.append(b["text"])
        elif hasattr(b, "text"):
            parts.append(b.text)
        else:
            parts.append(str(b))
    return "\n".join(parts)


def ask_model(client, model_id: str, history: list, user_text: str, system_text: str,
              max_tokens: int = 5000, temperature: float = 0.2) -> Tuple[List[dict], Any]:
    """Send `history` + one user turn.

    Returns (normalized single text block, raw response content).
    """
    resp = client.messages.create(
        model=model_id,
        system=system_text,
        messages=history + [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return [{"type": "text", "text": content_to_str(resp.content)}], resp.content
