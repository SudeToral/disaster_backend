import httpx
from langchain_ollama import ChatOllama
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


def get_llm(model: str | None = None) -> ChatOllama:
    """Factory: returns a configured ChatOllama instance."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=model or OLLAMA_MODEL,
        temperature=0,
        timeout=OLLAMA_TIMEOUT,
    )


async def is_ollama_available() -> bool:
    """Check if the Ollama server is reachable (async version)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def is_ollama_available_sync() -> bool:
    """Check if the Ollama server is reachable (sync version)."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
