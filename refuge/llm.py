"""Thin wrapper around the LLM backends.

Two providers are supported:

  * ``anthropic`` – Anthropic Messages API (default; the setting used in the paper)
  * ``openai``    – OpenAI Chat Completions API. With ``base_url`` this also covers any
                    OpenAI-compatible server (vLLM, SGLang, Ollama, Together, DeepSeek, ...),
                    so open-weight models can be plugged in as backbones.

The rest of the pipeline only calls ``ask_model`` and never touches the SDKs directly.
"""

from typing import Any, List, Optional, Tuple

SYSTEM_DS = "You are a data scientist for prediction task in relational database."
SYSTEM_JUDGE = "You are a rigorous judge for feature plan selection in relational databases."

PROVIDERS = ("anthropic", "openai")


def make_client(provider: str = "anthropic", api_key: Optional[str] = None,
                base_url: Optional[str] = None):
    """Build an SDK client. API keys default to $ANTHROPIC_API_KEY / $OPENAI_API_KEY."""
    kwargs = {k: v for k, v in (("api_key", api_key), ("base_url", base_url)) if v}
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(**kwargs)
    if provider == "openai":
        import openai
        return openai.OpenAI(**kwargs)
    raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")


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


def _is_openai_client(client) -> bool:
    return type(client).__module__.split(".")[0] == "openai"


# Models that rejected a non-default temperature (e.g. OpenAI reasoning models); remembered
# so the warning is printed and the failed request is paid for only once per model.
_NO_TEMPERATURE = set()


def _ask_openai(client, model_id, messages, system_text, max_tokens, temperature):
    import openai

    req = dict(
        model=model_id,
        messages=[{"role": "system", "content": system_text}]
        + [{"role": m["role"], "content": content_to_str(m["content"])} for m in messages],
        max_completion_tokens=max_tokens,
    )
    if model_id not in _NO_TEMPERATURE:
        try:
            return client.chat.completions.create(**req, temperature=temperature)
        except openai.BadRequestError as e:
            if "temperature" not in str(e):
                raise
            _NO_TEMPERATURE.add(model_id)
            print(f"[WARN] {model_id} does not accept temperature; using its default from now on.")
    return client.chat.completions.create(**req)


def ask_model(client, model_id: str, history: list, user_text: str, system_text: str,
              max_tokens: int = 5000, temperature: float = 0.2) -> Tuple[List[dict], Any]:
    """Send `history` + one user turn.

    Returns (normalized single text block, raw response content).
    """
    messages = history + [{"role": "user", "content": [{"type": "text", "text": user_text}]}]

    if _is_openai_client(client):
        resp = _ask_openai(client, model_id, messages, system_text, max_tokens, temperature)
        text = resp.choices[0].message.content or ""
        return [{"type": "text", "text": text}], text

    resp = client.messages.create(
        model=model_id,
        system=system_text,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return [{"type": "text", "text": content_to_str(resp.content)}], resp.content
