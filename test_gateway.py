"""What the gateway refuses before a model sees it. Run in the image, where FastAPI is:

    podman run --rm -v .:/app:ro -w /app --security-opt label=disable \\
        localhost/local-llm:latest python3 -m pytest -q -p no:cacheprovider test_gateway.py test_control.py
"""

import asyncio
import json
import os
import re
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

os.environ["LLM_BACKENDS"] = "tiny=http://backend"
os.environ["LLM_API_KEY"] = "sk-local"
os.environ["LLM_ADMIN_KEY"] = "sk-admin"
os.environ["LLM_OWNER_KEY"] = "sk-owner"
os.environ["LLM_KEYS_FILE"] = str(Path(tempfile.mkdtemp()) / "keys.json")

import gateway  # noqa: E402
from keys import Keys  # noqa: E402

KEY = {"authorization": "Bearer sk-local"}
ADMIN = {"x-api-key": "sk-admin"}
OWNER = {"x-api-key": "sk-owner"}
ASK = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}
WRONG_KEYS = [{}, {"x-api-key": "sk-nobody"}, {"authorization": "Bearer sk-loca"},
              {"authorization": "Bearer sk-local "}, {"authorization": "sk-local"}]
BUILT_IN = {"api": "sk-local", "admin": "sk-admin"}


@pytest.mark.parametrize("model", [None, [], {}, 1, True])
def test_invalid_model_types_are_client_errors_without_forwarding(web, backend, model):
    response = web.post("/v1/chat/completions", json=ASK | {"model": model}, headers=KEY)
    assert response.status_code == 400
    assert "string" in response.json()["error"]["message"]
    assert not backend


@pytest.fixture(autouse=True)
def keys(monkeypatch, tmp_path):
    """A fresh key list for every test, so one test's blocks stay out of the next."""
    fresh = Keys(tmp_path / "keys.json", BUILT_IN)
    monkeypatch.setattr(gateway, "keys", fresh)
    return fresh


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


def test_live_discovery_adds_catalog_capabilities_without_changing_standard_fields(web, monkeypatch):
    from model_catalog import MODELS

    capabilities = {name: model["capabilities"] for name, model in MODELS.items()}
    monkeypatch.setattr(gateway, "MODEL_CAPABILITIES", capabilities)
    standard = [{"id": name, "object": "model", "created": 123, "owned_by": "vllm",
                 "max_model_len": 32768} for name in capabilities]
    standard += [{"id": "unknown-model", "object": "model", "owned_by": "other", "max_model_len": 8192}]
    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": standard}))))
    response = web.get("/v1/models", headers=KEY)
    assert response.status_code == 200
    rows = response.json()["data"]
    assert [row["id"] for row in rows] == [row["id"] for row in standard]
    for original, enriched in zip(standard[:3], rows[:3]):
        assert enriched == original | {"capabilities": capabilities[original["id"]]}
    assert rows[-1] == standard[-1]
    assert rows[2]["capabilities"]["tools"] is False
    assert "sk-local" not in response.text and "sk-admin" not in response.text and "sk-owner" not in response.text


def test_catalog_metadata_does_not_advertise_offline_models_or_require_admin_access(web, monkeypatch):
    monkeypatch.setattr(gateway, "MODEL_CAPABILITIES", {"offline": {"tools": False}, "tiny": {"tools": True}})
    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"id": "tiny", "owned_by": "vllm"}]}))))
    assert web.get("/v1/models", headers=KEY).json() == {
        "object": "list", "data": [{"id": "tiny", "owned_by": "vllm", "capabilities": {"tools": True}}]}


def test_gateway_launch_passes_only_capabilities_from_cli_config(tmp_path):
    import shlex
    import subprocess

    source = Path(__file__).with_name("llm").read_text()
    function = re.search(r"start_gateway\(\) \{.*?\n\}", source, re.S).group(0)
    capture = tmp_path / "argv"
    config = {"models": [{"name": "known", "capabilities": {"tools": False, "input": ["text"]},
                           "private_setting": "synthetic-private"}, {"name": "legacy"}],
              "api_key": "synthetic-private-key"}
    program = "\n".join([
        "set -euo pipefail", f"config() {{ printf '%s' {shlex.quote(json.dumps(config))}; }}",
        "field() { cut -d'|' -f\"$2\" <<<\"$1\"; }", "remove() { :; }",
        "api_key() { echo synthetic-api; }", "admin_key() { echo synthetic-admin; }",
        "owner_key() { echo synthetic-owner; }", "url() { echo http://synthetic; }",
        f"podman_run() {{ printf '%s\\0' \"$@\" > {shlex.quote(str(capture))}; }}",
        'MODELS=("known|8007|0.23|owner/repo|")', "IMAGE=synthetic-image", "GATEWAY_PORT=8400",
        *[f"{name}={shlex.quote(str(tmp_path / name))}" for name in ("STATE_DIR", "CONTROL_DIR", "KEYS_DIR", "HERE")],
        function, "start_gateway"])
    subprocess.run(["bash", "-c", program], check=True, text=True, capture_output=True)
    arguments = capture.read_bytes().decode().split("\0")
    value = next(argument.partition("=")[2] for argument in arguments if argument.startswith("LLM_MODEL_CAPABILITIES="))
    assert json.loads(value) == {"known": {"tools": False, "input": ["text"]}}
    assert "synthetic-private" not in value


def test_missing_storage_reason_reaches_the_client_without_calling_the_backend(web, backend, monkeypatch):
    reason = "Required storage is not mounted at /mnt/archive"
    monkeypatch.setattr(gateway, "model_state", lambda model: {"state": "off", "reason": reason})
    response = web.post("/v1/chat/completions", json=ASK, headers=KEY)
    assert response.status_code == 503
    assert reason in response.json()["error"]["message"]
    assert backend == []


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


def test_a_client_that_hangs_up_stops_the_model(monkeypatch):
    """Without streaming, vLLM answers only when done, which a model caught in a
    loop may never be: the client leaving must stop it before then."""
    stopped = asyncio.Event()

    async def never_answers(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.sleep(3600)
        finally:
            stopped.set()

    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(transport=httpx.MockTransport(never_answers)))
    messages = [{"type": "http.request", "body": json.dumps(ASK).encode(), "more_body": False}]
    hang_up_at = time.monotonic() + 0.5  # the client waits a while, then gives up

    async def receive():
        # As uvicorn: once the client is gone, at once; until then, blocking.
        if messages:
            return messages.pop(0)
        if time.monotonic() < hang_up_at:
            await asyncio.sleep(hang_up_at - time.monotonic())
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
             "root_path": "", "query_string": b"", "client": ("127.0.0.1", 50000), "server": ("gateway", 8400),
             "headers": [(b"authorization", b"Bearer sk-local"), (b"content-type", b"application/json")]}

    async def scenario():
        await asyncio.wait_for(gateway.app(scope, receive, send), timeout=10)
        await asyncio.wait_for(stopped.wait(), timeout=1)

    asyncio.run(scenario())


def test_the_owner_hands_out_a_key_and_blocks_it(web, backend):
    made = web.post("/owner/keys", json={"name": "sam", "role": "admin"}, headers=OWNER)
    assert made.status_code == 201
    sams = {"x-api-key": made.json()["key"]}
    assert web.post("/v1/chat/completions", json=ASK, headers=sams).status_code == 200
    assert web.get("/admin/models", headers=sams).status_code == 202

    assert web.post("/owner/keys/sam/block", headers=OWNER).json()["blocked"]
    assert web.post("/v1/chat/completions", json=ASK, headers=sams).status_code == 403
    assert web.get("/admin/models", headers=sams).status_code == 403
    assert web.post("/v1/chat/completions", json=ASK, headers=ADMIN).status_code == 200  # others still work

    web.post("/owner/keys/sam/unblock", headers=OWNER)
    assert web.post("/v1/chat/completions", json=ASK, headers=sams).status_code == 200


def test_the_shared_admin_key_can_be_blocked_too(web, backend):
    web.post("/owner/keys/admin/block", headers=OWNER)
    assert web.post("/v1/chat/completions", json=ASK, headers=ADMIN).status_code == 403
    assert web.post("/v1/chat/completions", json=ASK, headers=KEY).status_code == 200


def test_a_block_survives_a_restart(web, keys):
    web.post("/owner/keys/admin/block", headers=OWNER)
    assert Keys(keys.path, BUILT_IN).entries["admin"]["blocked"]


def test_replacing_a_shared_key_file_starts_it_unblocked(web, keys):
    web.post("/owner/keys/admin/block", headers=OWNER)
    restarted = Keys(keys.path, BUILT_IN | {"admin": "sk-admin-new"})
    assert not restarted.entries["admin"]["blocked"]


@pytest.mark.parametrize("headers", [KEY, ADMIN])
def test_only_the_owner_key_manages_keys(web, headers):
    assert web.get("/owner/keys", headers=headers).status_code == 403
    assert web.post("/owner/keys/admin/block", headers=headers).status_code == 403
    assert web.post("/owner/keys", json={"name": "me", "role": "admin"}, headers=headers).status_code == 403


def test_the_owner_key_is_refused_over_plain_http_from_the_network(neighbour, keys):
    assert neighbour.post("/owner/keys/admin/block", headers=OWNER).status_code == 403
    assert not keys.entries["admin"]["blocked"]


def test_the_owner_sees_who_used_which_key(web, backend):
    web.post("/v1/chat/completions", json=ASK, headers=ADMIN | {"x-forwarded-for": "203.0.113.9"})
    admin = next(key for key in web.get("/owner/keys", headers=OWNER).json() if key["name"] == "admin")
    assert admin["key"] == "sk-admin"
    assert admin["requests"] == 1
    assert admin["last_from"] == "203.0.113.9"


def test_names_are_unique_and_shared_keys_cannot_be_deleted(web):
    assert web.post("/owner/keys", json={"name": "admin", "role": "api"}, headers=OWNER).status_code == 409
    assert web.post("/owner/keys", json={"name": "a/b", "role": "api"}, headers=OWNER).status_code == 409
    assert web.delete("/owner/keys/admin", headers=OWNER).status_code == 409
    web.post("/owner/keys", json={"name": "sam", "role": "api"}, headers=OWNER)
    assert web.delete("/owner/keys/sam", headers=OWNER).status_code == 200
    assert "sam" not in {key["name"] for key in web.get("/owner/keys", headers=OWNER).json()}
