"""Shared Groq client setup used by the router, sql_agent, and rag_agent."""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"  # ponytail: Groq deprecated llama-3.3-70b-versatile; swap if it disappears too

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = Groq(api_key=api_key)
    return _client


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()
