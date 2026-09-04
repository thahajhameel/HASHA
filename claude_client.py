"""
claude_client.py — thin wrapper around the Anthropic SDK.

Every call here asks Claude to return ONLY JSON, then parses it defensively
(stripping markdown fences if the model adds them anyway). Centralizing this
means every endpoint in app.py gets consistent error handling.
"""

import os
import json
import re
from anthropic import Anthropic

MODEL = os.environ.get("HASHA_MODEL", "claude-sonnet-5")

_client = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def call_json(system: str, user: str, max_tokens: int = 1000) -> dict | None:
    """Call Claude and parse the response as JSON. Returns None on failure
    so callers can decide how to handle/report a bad generation."""
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return _parse_json(text)


def _parse_json(text: str) -> dict | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    if start == -1:
        start = cleaned.find("[")
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
