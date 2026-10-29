"""What the gateway refuses before a model sees it. Run in the image, where FastAPI is:

    podman run --rm -v .:/app:ro -w /app --security-opt label=disable \\
        localhost/local-llm:latest python3 -m pytest -q -p no:cacheprovider test_gateway.py test_control.py
"""

import json
import os
import re

import httpx
import pytest
from fastapi.testclient import TestClient

os.environ["LLM_BACKENDS"] = "tiny=http://backend"
os.environ["LLM_API_KEY"] = "sk-owner"
os.environ["LLM_ADMIN_KEY"] = "sk-admin"

import gateway  # noqa: E402

KEY = {"authorization": "Bearer sk-owner"}
ADMIN = {"x-api-key": "sk-admin"}
ASK = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}
WRONG_KEYS = [{}, {"x-api-key": "sk-nobody"}, {"authorization": "Bearer sk-owne"},
              {"authorization": "Bearer sk-owner "}, {"authorization": "sk-owner"}]


@pytest.fixture
def backend(monkeypatch):
    """A stand-in for vLLM that records the requests it is sent."""
    seen = []

    async def chunks():
        yield b'{"ok": true}'

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=chunks())  # streamed, as vLLM answers

    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    return seen


@pytest.fixture
def control(monkeypatch):
    """A stand-in for control.py that records the requests it is sent."""
    seen = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json={"model": "tiny"})

    monkeypatch.setattr(gateway, "control", httpx.AsyncClient(
        transport=httpx.MockTransport(answer), base_url="http://control"))
    return seen


@pytest.fixture
def web(backend, control):
    """Requests from this computer, as through Caddy."""
    return TestClient(gateway.app, client=("127.0.0.1", 50000))


@pytest.fixture
def neighbour(backend, control):
    """Plain HTTP from another computer in the network."""
    return TestClient(gateway.app, client=("192.168.178.20", 50000))


def test_request_with_the_key_reaches_the_model(web, backend):
    assert web.post("/v1/chat/completions", json=ASK, headers=KEY).json() == {"ok": True}
    assert [json.loads(request.content) for request in backend] == [ASK]


def test_the_key_is_not_passed_on_to_the_model(web, backend):
    web.post("/v1/messages", json=ASK, headers=KEY | ADMIN)
    assert not {"authorization", "x-api-key"} & set(backend[0].headers)


@pytest.mark.parametrize("headers", WRONG_KEYS)
def test_request_without_the_key_is_turned_away(web, backend, headers):
    assert web.post("/v1/chat/completions", json=ASK, headers=headers).status_code == 401
    assert web.get("/stats", headers=headers).status_code == 401
    assert backend == []


def every_route():
    for route in gateway.app.routes:
        path = re.sub(r"\{(\w+)(:\w+)?\}", "tiny", route.path)
        for method in route.methods:
            yield method, path


@pytest.mark.parametrize("method, path", sorted(every_route()))
def test_every_route_needs_a_key_unless_public(web, backend, control, method, path):
    """A route added later is closed until it is put in PUBLIC on purpose."""
    status = web.request(method, path).status_code
    if path in gateway.PUBLIC:
        assert status != 401
    else:
        assert status == 401
        assert backend == control == []


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/admin", "/nothing", "/v1", "/health/", "//v1/models"])
def test_paths_that_match_no_route_need_a_key_too(web, path):
    assert web.get(path).status_code == 401


def test_starting_and_stopping_takes_the_admin_key(web, control):
    assert web.post("/admin/models/tiny/start", headers=KEY).status_code == 403
    assert web.get("/admin/models", headers=KEY).status_code == 403
    assert control == []
    assert web.post("/admin/models/tiny/start?replace=true", headers=ADMIN).status_code == 202
    assert str(control[0].url) == "http://control/models/tiny/start?replace=true"


def test_the_admin_key_is_refused_over_plain_http_from_the_network(neighbour, control):
    assert neighbour.get("/admin/models", headers=ADMIN).status_code == 403
    assert neighbour.post("/admin/models/tiny/stop", headers=ADMIN | {"x-forwarded-for": "127.0.0.1"}).status_code == 403
    assert control == []


def test_the_admin_key_works_over_https_from_anywhere(backend, control):
    through_caddy = TestClient(gateway.app, base_url="https://llm.example", client=("203.0.113.9", 50000))
    assert through_caddy.get("/admin/models", headers=ADMIN).status_code == 202


def test_the_api_key_still_works_from_the_network(neighbour, backend):
    assert neighbour.post("/v1/chat/completions", json=ASK, headers=KEY).status_code == 200


def test_only_the_pages_own_script_may_run(web):
    response = web.get("/")
    policy = response.headers["content-security-policy"]
    nonce = policy.split("'nonce-")[1].split("'")[0]
    assert "unsafe-inline" not in policy.split("script-src")[1].split(";")[0]
    assert f'<script nonce="{nonce}">' in response.text
    assert nonce not in web.get("/").text  # fresh every time


def test_the_admin_key_also_reaches_the_models(web, backend):
    assert web.post("/v1/chat/completions", json=ASK, headers=ADMIN).status_code == 200


# Encoded, because the test client would resolve plain dots before sending;
# the server decodes them, so the gateway sees ".." either way.
@pytest.mark.parametrize("path", ["%2e%2e/health", "a/%2e%2e/%2e%2e/tokenize", "%2e/chat/completions"])
def test_paths_outside_v1_are_not_reachable_even_with_the_key(web, backend, path):
    assert web.post(f"/v1/{path}", json=ASK, headers=KEY).status_code == 404
    assert backend == []


def test_oversized_request_is_refused(web, backend, monkeypatch):
    monkeypatch.setattr(gateway, "LARGEST_BODY", 1000)
    big = ASK | {"padding": "x" * 2000}
    assert web.post("/v1/chat/completions", json=big, headers=KEY).status_code == 413
    assert backend == []
