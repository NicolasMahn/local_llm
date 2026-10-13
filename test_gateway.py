"""What the gateway refuses before a model sees it. Run in the image, where FastAPI is:

    podman run --rm -v .:/app:ro -w /app --security-opt label=disable \\
        localhost/local-llm:latest python3 -m pytest -q -p no:cacheprovider test_gateway.py
"""

import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

os.environ["LLM_BACKENDS"] = "tiny=http://backend"
os.environ["LLM_API_KEY"] = "sk-owner"

import gateway  # noqa: E402

KEY = {"authorization": "Bearer sk-owner"}
ASK = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def backend(monkeypatch):
    """A stand-in for vLLM that records the requests it is sent."""
    seen = []

    async def chunks():
        yield b'{"ok": true}'

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=chunks())  # streamed, as vLLM answers

    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    return seen


@pytest.fixture
def web(backend):
    return TestClient(gateway.app)


def test_request_with_the_key_reaches_the_model(web, backend):
    assert web.post("/v1/chat/completions", json=ASK, headers=KEY).json() == {"ok": True}
    assert backend == [ASK]


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "sk-nobody"}, {"authorization": "Bearer sk-owne"}])
def test_request_without_the_key_is_turned_away(web, backend, headers):
    assert web.post("/v1/chat/completions", json=ASK, headers=headers).status_code == 401
    assert web.get("/stats", headers=headers).status_code == 401
    assert backend == []


def test_oversized_request_is_refused(web, backend, monkeypatch):
    monkeypatch.setattr(gateway, "LARGEST_BODY", 1000)
    big = ASK | {"padding": "x" * 2000}
    assert web.post("/v1/chat/completions", json=big, headers=KEY).status_code == 413
    assert backend == []
