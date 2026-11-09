"""One port in front of several vLLM servers.

vLLM already speaks both the OpenAI and the Anthropic API, so the gateway
only has to check the key and pick a backend by the request's "model" field.
It also serves a small test chat page at / for other devices on the network,
at /stats what monitor.py last measured, for that page's graphs, and at
/admin/ the control service (control.py), which starts and stops models.

Every path needs a key unless it is in PUBLIC. Keys come in three roles: an
API key, which sits in clients, reaches the models and the measurements; an
admin key also starts and stops models at /admin/; the owner key also sees,
makes and blocks the others at /owner/ (keys.py).
"""

import json
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from keys import Keys

# "name=http://host:port,name=http://host:port" — set by the llm script.
BACKENDS = dict(
    pair.split("=", 1) for pair in os.environ["LLM_BACKENDS"].split(",")
)
OWNER_KEY = os.environ["LLM_OWNER_KEY"]
keys = Keys(Path(os.environ.get("LLM_KEYS_FILE", "/keys/keys.json")),
            {"api": os.environ["LLM_API_KEY"], "admin": os.environ["LLM_ADMIN_KEY"]})
STATS = Path("/state/stats.json")
CONTROL = "/control/control.sock"
# The page holds no key, it asks the visitor for one; /health says only
# that the gateway answers, for the monitor.
PUBLIC = {"/", "/health"}
# No request is read into memory beyond this; the chat page's largest
# attachment is 50 MB, which base64 grows by a third.
LARGEST_BODY = 100 * 1024**2

# No generated API docs: nothing should be reachable that PUBLIC does not list.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
client = httpx.AsyncClient(timeout=None)
control = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=CONTROL), base_url="http://control",
                            timeout=30)


def model_state(model: str) -> dict | None:
    """What monitor.py last saw of a model, or None when its snapshot is missing or stale."""
    try:
        stats = json.loads(STATS.read_text())
        age = time.time() - datetime.fromisoformat(stats["updated"]).timestamp()
    except (OSError, ValueError, KeyError):
        return None
    return stats.get("model_states", {}).get(model) if age < 15 else None


def not_live(model: str) -> JSONResponse | None:
    """A clear answer for a model that is not live, instead of a failed connection."""
    state = model_state(model)
    if state is None or state["state"] == "live":
        return None  # Unknown or fine: let the request through and see.
    if state["state"] == "starting":
        minutes = round((time.time() - datetime.fromisoformat(state["started"]).timestamp()) / 60)
        response = error(503, f"{model} is still starting ({minutes} min so far).")
        response.headers["retry-after"] = "30"
        return response
    if state["state"] == "failed":
        return error(503, f"{model} crashed: {state['reason']}")
    return error(503, f"{model} is not running; start it with ./llm up {model} or the panel.")


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


def sent_key(request: Request) -> str:
    # OpenAI clients send a bearer token, Anthropic clients an x-api-key header.
    scheme, _, bearer = request.headers.get("authorization", "").partition(" ")
    return request.headers.get("x-api-key") or (bearer if scheme.lower() == "bearer" else "")


def is_key(sent: str, key: str) -> bool:
    return secrets.compare_digest(sent.encode(), key.encode())


@app.middleware("http")
async def require_key(request: Request, call_next):
    # The path the router matches, so the check and the route cannot disagree.
    path = request.scope["path"]
    if path in PUBLIC:
        return await call_next(request)
    role = role_of(request)
    if role is None:
        return error(401, "Missing or wrong API key, see ./llm key")
    if role == "blocked":
        return error(403, "This key has been blocked.")
    area = path.split("/")[1]  # "admin" for /admin/models
    if area == "owner" and role != "owner":
        return error(403, "Managing keys takes the owner key, see ./llm owner-key")
    if area == "admin" and role == "api":
        return error(403, "Starting and stopping models takes an admin key, see ./llm admin-key")
    if area in ("admin", "owner"):
        # Plain HTTP from another computer crossed the network readable by anyone
        # on it.
        if not encrypted_or_local(request):
            return error(403, f"The {role} key works over the HTTPS address only (./llm url); "
                         "it crossed the network unencrypted, so consider replacing it")
    return await call_next(request)


def role_of(request: Request) -> str | None:
    """owner, admin, api or blocked for the key sent, None for no known key."""
    sent = sent_key(request)
    if is_key(sent, OWNER_KEY):
        return "owner"
    name = keys.find(sent)
    if name is None:
        return None
    if keys.entries[name]["blocked"]:
        return "blocked"
    keys.used(name, address(request))
    return keys.entries[name]["role"]


def address(request: Request) -> str | None:
    """Where the request came from; Caddy, on this computer, says so for the internet."""
    if request.client is None:
        return None
    if request.client.host in ("127.0.0.1", "::1"):
        return request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    return request.client.host


def encrypted_or_local(request: Request) -> bool:
    # HTTPS ends at Caddy, here, which says so in X-Forwarded-Proto; uvicorn
    # believes that header only from 127.0.0.1, so no one else can claim it.
    local = request.client is not None and request.client.host in ("127.0.0.1", "::1")
    return request.url.scheme == "https" or local


async def read_body(request: Request) -> bytes | None:
    """The request body, or None when it is larger than LARGEST_BODY."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > LARGEST_BODY:
            return None
    return bytes(body)


class Forwarded(StreamingResponse):
    """A backend's answer, passed on as it arrives.

    Closes the backend's side here rather than in a background task, which is
    skipped when the client hangs up mid-answer: the model must stop writing
    then.
    """

    def __init__(self, upstream: httpx.Response):
        super().__init__(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )
        self.upstream = upstream

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.upstream.aclose()


# The page keeps the keys in the browser. Only its own script runs (it carries
# a fresh nonce), so a slip in its escaping of model output cannot become
# script; whatever does run can send nothing anywhere but here; and no other
# site may frame the page to trick a click on its start and stop buttons.
PAGE_POLICY = ("default-src 'self'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
               "img-src 'self' data: blob:; media-src 'self' data: blob:; "
               "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


@app.get("/")
async def chat_page():
    # Read on every request, so an edited page is served without a restart.
    nonce = secrets.token_urlsafe(16)
    page = Path(__file__).with_name("chat.html").read_text().replace("<script>", f'<script nonce="{nonce}">')
    return HTMLResponse(page, headers={
        "content-security-policy": PAGE_POLICY.format(nonce=nonce), "referrer-policy": "no-referrer",
        "cache-control": "no-store",
    })


@app.get("/stats")
async def stats():
    if not STATS.exists():
        return error(503, "Nothing measured yet: the monitor is not running, see ./llm install")
    return FileResponse(STATS, media_type="application/json")


@app.get("/v1/models")
async def list_models():
    models = []
    for url in BACKENDS.values():
        try:
            response = await client.get(f"{url}/v1/models", timeout=5)
            models += response.json()["data"]
        except (httpx.HTTPError, KeyError, ValueError):
            pass  # A backend that is still loading just isn't listed yet.
    return {"object": "list", "data": models}


@app.get("/health")
async def health():
    return {"status": "ok"}


async def to_control(method: str, path: str, **options) -> JSONResponse:
    try:
        response = await control.request(method, path, **options)
        return JSONResponse(response.json(), status_code=response.status_code)
    except (httpx.HTTPError, ValueError):
        return error(503, "The control service is not running, see ./llm install")


@app.get("/admin/models")
async def admin_models():
    return await to_control("GET", "/models")


@app.post("/admin/models/{name}/{action}")
async def admin_model(name: str, action: Literal["start", "stop", "restart"], request: Request):
    if name not in BACKENDS:
        return error(404, f"Unknown model {name!r}. Available: {sorted(BACKENDS)}")
    return await to_control("POST", f"/models/{name}/{action}", params=request.query_params)


class NewKey(BaseModel):
    name: str
    role: Literal["api", "admin"]


@app.get("/owner/keys")
async def list_keys():
    return keys.listing()


@app.post("/owner/keys", status_code=201)
async def create_key(new: NewKey):
    try:
        return keys.create(new.name, new.role)
    except ValueError as exc:
        return error(409, str(exc))


@app.post("/owner/keys/{name}/{action}")
async def block_key(name: str, action: Literal["block", "unblock"]):
    if name not in keys.entries:
        return error(404, f"No key named {name!r}.")
    return keys.block(name, action == "block")


@app.delete("/owner/keys/{name}")
async def delete_key(name: str):
    if name not in keys.entries:
        return error(404, f"No key named {name!r}.")
    try:
        keys.delete(name)
    except ValueError as exc:
        return error(409, str(exc))
    return {"deleted": name}


@app.post("/v1/{path:path}")
async def proxy(path: str, request: Request):
    # httpx resolves dot segments, so /v1/../x would reach vLLM's endpoints
    # outside /v1 (cache resets, tokenizer, metrics) that clients have no use for.
    if any(part in (".", "..") for part in path.split("/")):
        return error(404, "Not found")
    body = await read_body(request)
    if body is None:
        return error(413, f"Requests can be up to {LARGEST_BODY // 1024**2} MB.")

    try:
        model = json.loads(body).get("model")
    except (ValueError, AttributeError):
        return error(400, "Request body must be a JSON object.")
    if model not in BACKENDS:
        return error(404, f"Unknown model {model!r}. Available: {sorted(BACKENDS)}")
    if refusal := not_live(model):
        return refusal

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in ("host", "content-length", "authorization", "x-api-key")
    }
    try:
        upstream = await client.send(
            client.build_request(
                "POST", f"{BACKENDS[model]}/v1/{path}", content=body, headers=headers,
                params=request.query_params,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        return error(502, f"{model} is not answering: {exc}")
    return Forwarded(upstream)
