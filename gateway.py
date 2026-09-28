"""One port in front of several vLLM servers.

vLLM already speaks both the OpenAI and the Anthropic API, so the gateway
only has to check the key and pick a backend: by the request's "model" field,
or by path for audio uploads (multipart bodies have no JSON "model" to read).
It also serves a small test chat page at / for other devices on the network,
and at /stats what monitor.py last measured, for that page's graphs.
"""

import json
import os
import secrets
import time
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# "name=http://host:port,name=http://host:port" — set by the llm script.
BACKENDS = dict(
    pair.split("=", 1) for pair in os.environ["LLM_BACKENDS"].split(",")
)
AUDIO_MODEL = os.environ["LLM_AUDIO_MODEL"]
API_KEY = os.environ["LLM_API_KEY"]
STATS = Path("/state/stats.json")

app = FastAPI()
client = httpx.AsyncClient(timeout=None)


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
    bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
    return request.headers.get("x-api-key") or bearer


@app.middleware("http")
async def require_key(request: Request, call_next):
    protected = request.url.path.startswith("/v1/") or request.url.path == "/stats"
    if protected and not secrets.compare_digest(
        sent_key(request).encode(), API_KEY.encode()
    ):
        return error(401, "Missing or wrong API key, see ./llm key")
    return await call_next(request)


@app.get("/")
async def chat_page():
    # Public on purpose: the page holds no key, it asks the visitor for one.
    return FileResponse(Path(__file__).with_name("chat.html"))


@app.get("/stats")
async def stats():
    if not STATS.exists():
        return error(503, "Nothing measured yet: the monitor is not running, see ./llm monitor")
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
    status = {}
    for name, url in BACKENDS.items():
        try:
            ok = (await client.get(f"{url}/health", timeout=5)).is_success
        except httpx.HTTPError:
            ok = False
        status[name] = "up" if ok else "down"
    return status


@app.post("/v1/{path:path}")
async def proxy(path: str, request: Request):
    body = await request.body()

    if path.startswith("audio/"):
        model = AUDIO_MODEL
    else:
        try:
            model = json.loads(body).get("model")
        except ValueError:
            return error(400, "Request body must be JSON.")
        if model not in BACKENDS:
            return error(404, f"Unknown model {model!r}. Available: {sorted(BACKENDS)}")
    if refusal := not_live(model):
        return refusal

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in ("host", "content-length")
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

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        background=upstream.aclose,
    )
