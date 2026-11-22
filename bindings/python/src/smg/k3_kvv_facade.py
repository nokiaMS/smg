"""K3 KVV router protocol facade: public admission + response formatting.

Runs in front of the unmodified SMG 1.12.0 Rust gateway (smg) inside the
router pod:

    client -> facade :8000 (public) -> smg :8300 (loopback, private)

Responsibilities (K3 KVV customer contract, protocol_cache_bench 2.7.0):

1. Admission (HTTP 400, structured ``error`` object naming the field):
   - /v1/messages            top-level ``cache_control``    {type, ttl}
   - /v1/responses           top-level ``prompt_cache_options`` {mode, ttl}
   - /v1/chat/completions    top-level ``prompt_cache_options`` {mode, ttl}
   - chat ``tool_choice`` object form is refused while thinking is not
     explicitly disabled (Kimi specified-tool / thinking conflict).

2. Retention hint to the workers: every request that passes admission with a
   TTL hint carries the private header ``x-sglang-cache-retention-ttl``
   (decimal seconds) to SMG, which forwards allow-listed headers verbatim.
   Unsolicited client-supplied private ``x-sglang-*`` headers are stripped
   before forwarding and never answered publicly.

3. Response formatting from the decode receipt: when the decode worker answers
   with the private header ``x-sglang-prompt-cache-receipt`` = ``R,W,A``
   (cached input tokens, actually inserted prompt-cache tokens, approved
   retention seconds), the facade maps it onto the protocol's public usage
   schema, adds the Kimi ``msh-usage-cache-*`` vendor headers, and (Responses
   only) echoes the applied ``prompt_cache_options`` on the terminal event.
   A measured zero is valid and forwarded; a missing receipt is never guessed
   as zero — the request simply keeps the worker's own accounting.
   Non-opt-in paths never receive Msh headers from this facade.

Everything else (bodies, SSE payloads, status codes, headers) is forwarded
byte-for-byte in both directions.
"""

from __future__ import annotations

import json
import os
import re

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Contract constants (source: customer protocol_cache_bench 2.7.0:
# pcc_cache_ttl.HEADER_NAMES, pcc_cache_validation.modes_for/target_for,
# pcc._CACHE_TTL_RE equivalent, Kimi platform docs for messages/responses/chat).
# ---------------------------------------------------------------------------

TTL_RE = re.compile(r"^([1-9][0-9]*(?:\.[0-9]+)?)(s|m|h)$")
TTL_UNITS = {"s": 1, "m": 60, "h": 3600}
VALID_TTL_BUCKETS = ("5m", "1h")  # customer's cache_ttl_values; the regex is the law

MSH_WRITE_HEADER = "msh-usage-cache-write-tokens-{ttl}"
MSH_READ_HEADER = "msh-usage-cache-read-tokens"

# Private wire with the worker middleware (owned by Main/WorkerK3):
HINT_HEADER = "x-sglang-cache-retention-ttl"
RECEIPT_HEADER = "x-sglang-prompt-cache-receipt"

# The upstream SMG gateway: loopback only, never directly reachable.
# Ports come from the launch_router overlay via env, with the same defaults.
UPSTREAM_HOST = os.environ.get("K3_KVV_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("K3_KVV_UPSTREAM_PORT", "8300"))
UPSTREAM = f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}"

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length", "host",
}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(600.0, connect=30.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ttl_seconds(ttl: str) -> int | None:
    """Integer seconds, guarded: the worker accepts only an integer regex,
    so a float overflow or an absurd magnitude is refused upstream (None)."""
    m = TTL_RE.match(ttl)
    if not m:
        return None
    try:
        seconds = int(float(m.group(1)) * TTL_UNITS[m.group(2)])
    except (ValueError, OverflowError):
        return None
    if seconds <= 0 or seconds > 10**9:
        return None
    return seconds


def _bad_request(param: str, message: str) -> JSONResponse:
    """400 shaped like the customer's rejection evidence expects: a JSON
    object with a structured ``error`` (non-empty message; ``param`` names
    the offending field so `cache_validation_require_field_name` passes)."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_cache_parameter",
                "message": message,
                "param": param,
            }
        },
    )


def _validate_cache_options(body: dict, key: str, kind_key: str, valid_kind: str) -> JSONResponse | None:
    """Shared admission for the two cache-option containers.

    ``key`` is the top-level body field (``cache_control`` or
    ``prompt_cache_options``), ``kind_key`` the discriminator inside it
    (``type`` or ``mode``). Mirrors pcc_cache_validation.request_body: only
    ``type=ephemeral`` / ``mode=implicit`` and a ``<n><s|m|h>`` ttl pass.
    """
    options = body.get(key)
    if options is None:
        return None
    if not isinstance(options, dict):
        return _bad_request(key, f"'{key}' must be an object")
    kind = options.get(kind_key)
    if kind is not None and kind != valid_kind:
        return _bad_request(
            f"{key}.{kind_key}",
            f"Invalid value for '{key}.{kind_key}': must be '{valid_kind}', got '{kind}'",
        )
    ttl = options.get("ttl")
    if ttl is not None:
        if not isinstance(ttl, str) or TTL_RE.match(ttl) is None or _ttl_seconds(ttl) is None:
            return _bad_request(
                f"{key}.ttl",
                f"Invalid value for '{key}.ttl': must look like '90s', '5m' or '1h'",
            )
    return None


def _messages_admission(body: dict) -> JSONResponse | None:
    return _validate_cache_options(body, "cache_control", "type", "ephemeral")


def _options_admission(body: dict) -> JSONResponse | None:
    return _validate_cache_options(body, "prompt_cache_options", "mode", "implicit")


def _chat_tool_choice_admission(body: dict) -> JSONResponse | None:
    """Kimi node-5 contract: ``tool_choice`` object form (specified tool) is
    incompatible with thinking; the reverse contract expects HTTP 400 naming
    both parameters (HR signature: message mentions tool_choice AND thinking
    plus an incompatibility word, no explicit field location)."""
    tool_choice = body.get("tool_choice")
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return None
    thinking = body.get("thinking")
    disabled = isinstance(thinking, dict) and thinking.get("type") == "disabled"
    if disabled:
        return None
    return _bad_request(
        "tool_choice",
        "Specified tool_choice is incompatible with thinking mode: "
        "'tool_choice' of type 'function' is not supported while thinking is enabled",
    )


def _extract_hint(body: dict, route: str) -> tuple[str | None, int | None]:
    """Return (requested ttl string, ttl seconds) for an opt-in request.

    A present container with no ``ttl`` opts in at the vendor default 5m
    (messages ``cache_control={"type":"ephemeral"}``, responses/chat
    ``prompt_cache_options={"mode":"implicit"}``) — the same default the
    worker applies, so the hint channel stays consistent.
    """
    if route == "/v1/messages":
        options = body.get("cache_control")
    else:
        options = body.get("prompt_cache_options")
    if not isinstance(options, dict):
        return None, None
    ttl = options.get("ttl")
    if ttl is None:
        return "5m", 300
    if isinstance(ttl, str) and TTL_RE.match(ttl):
        seconds = _ttl_seconds(ttl)
        if seconds is not None:
            return ttl, seconds
    return None, None


def _clean_request_headers(headers) -> dict[str, str]:
    """Forward request headers minus hop-by-hop and any client-supplied
    private ``x-sglang-*`` markers: only the facade may mint those."""
    out = {}
    for name, value in headers.items():
        low = name.lower()
        if low in HOP_BY_HOP or low.startswith("x-sglang-"):
            continue
        out[name] = value
    return out


def _parse_receipt(value: str | None) -> tuple[int, int, int] | None:
    """Decode ``R,W,A`` decimal receipt; None when absent or malformed."""
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 3:
        return None
    try:
        r, w, a = (int(p.strip()) for p in parts)
    except ValueError:
        return None
    if r < 0 or w < 0 or a < 0:
        return None
    return r, w, a


def _clean_response_headers(headers) -> dict[str, str]:
    """Response headers minus hop-by-hop (incl. content-encoding: httpx has
    already decoded the body, so keeping it would mislabel the bytes), the
    private receipt marker and SMG's own error-code header; the facade
    re-issues the public Msh evidence itself."""
    out = {}
    for name, value in headers.items():
        low = name.lower()
        if low in HOP_BY_HOP or low == RECEIPT_HEADER or low == "x-smg-error-code":
            continue
        out[name] = value
    return out


def _applied_bucket(approved_s: int) -> str | None:
    """K3.1 CacheMeter::msh_split: actual approved retention decides the
    bucket — 1..300s -> 5m, >300s -> 1h, 0 (denied) -> none (W stays
    protected, misbilling impossible)."""
    if 0 < approved_s <= 300:
        return "5m"
    if approved_s > 300:
        return "1h"
    return None


def _receipt_to_msh(receipt: tuple[int, int, int]) -> dict[str, str]:
    """Vendor evidence headers per pcc_cache_ttl.HEADER_NAMES. Both write
    buckets are always present; the approved-retention bucket carries W,
    the other a measured zero. Read evidence always present."""
    read, write, approved = receipt
    headers = {MSH_READ_HEADER: str(read)}
    applied = _applied_bucket(approved)
    for bucket in VALID_TTL_BUCKETS:
        headers[MSH_WRITE_HEADER.format(ttl=bucket)] = str(write if bucket == applied else 0)
    return headers


def _receipt_to_usage_patch(receipt: tuple[int, int, int], route: str) -> dict:
    """Protocol-native usage fields for the receipt.

    - messages: ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
      plus the ``cache_creation.ephemeral_<bucket>_input_tokens`` split.
    - chat: ``prompt_tokens_details.cache_write_tokens`` / ``.cached_tokens``.
    - responses: ``input_tokens_details.cache_write_tokens`` / ``.cached_tokens``.
    """
    read, write, _ = receipt
    if route == "/v1/messages":
        return {
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write,
        }
    details = {"cache_write_tokens": write, "cached_tokens": read, "cache_write_ttl_s": receipt[2]}
    if route == "/v1/chat/completions":
        return {"prompt_tokens_details": details}
    return {"input_tokens_details": details}


def _applied_options(requested_ttl, approved_s):
    if not approved_s:
        return None
    ttl = requested_ttl if requested_ttl and _ttl_seconds(requested_ttl) == approved_s else f"{approved_s}s"
    return {"mode": "implicit", "ttl": ttl}


def _deep_merge_usage(base: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_usage(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Response rewriting (non-streaming)
# ---------------------------------------------------------------------------

def _apply_receipt_json(raw: bytes, receipt: tuple[int, int, int], route: str, requested_ttl: str | None) -> bytes:
    """Merge the receipt into the JSON usage object; add the per-bucket
    ``cache_creation.ephemeral_<ttl>_input_tokens`` split for Messages."""
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw
    if not isinstance(obj, dict) or isinstance(obj.get("error"), dict):
        return raw
    usage = obj.setdefault("usage", {})
    if not isinstance(usage, dict):
        return raw
    _deep_merge_usage(usage, _receipt_to_usage_patch(receipt, route))
    if route == "/v1/messages":
        split = usage.setdefault("cache_creation", {})
        if isinstance(split, dict):
            applied = _applied_bucket(receipt[2])
            for bucket in VALID_TTL_BUCKETS:
                split[f"ephemeral_{bucket}_input_tokens"] = receipt[1] if bucket == applied else 0
    elif route == "/v1/responses":
        obj["prompt_cache_options"] = _applied_options(requested_ttl, receipt[2])
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Response rewriting (streaming): the SSE relay patches only the accounting
# payloads (chat final usage chunk / messages message_start + message_delta /
# responses terminal event) and appends nothing else.
# ---------------------------------------------------------------------------



def _patch_sse_line(line: str, receipt, route, requested_ttl, state: dict) -> str:
    """Rewrite one SSE line's JSON data when it carries the protocol's
    usage accounting. Returns the original line when anything is off."""
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return line
    payload = stripped[5:].strip()
    if not payload or payload == "[DONE]":
        return line
    try:
        obj = json.loads(payload)
    except ValueError:
        return line
    if not isinstance(obj, dict) or isinstance(obj.get("error"), dict):
        return line
    changed = False
    if route == "/v1/chat/completions":
        if isinstance(obj.get("usage"), dict) and obj.get("choices") in (None, []):
            _deep_merge_usage(obj["usage"], _receipt_to_usage_patch(receipt, route))
            changed = True
    elif route == "/v1/messages":
        kind = obj.get("type")
        if kind == "message_start" and isinstance(obj.get("message"), dict) and not state.get("start_done"):
            message = obj["message"]
            usage = message.setdefault("usage", {})
            if isinstance(usage, dict):
                _deep_merge_usage(usage, _receipt_to_usage_patch(receipt, route))
                split = usage.setdefault("cache_creation", {})
                if isinstance(split, dict):
                    applied = _applied_bucket(receipt[2])
                    for bucket in VALID_TTL_BUCKETS:
                        split[f"ephemeral_{bucket}_input_tokens"] = (
                            receipt[1] if bucket == applied else 0
                        )
                changed = True
                state["start_done"] = True
        elif kind == "message_delta" and isinstance(obj.get("usage"), dict):
            _deep_merge_usage(obj["usage"], _receipt_to_usage_patch(receipt, route))
            split = obj["usage"].setdefault("cache_creation", {})
            if isinstance(split, dict):
                applied = _applied_bucket(receipt[2])
                for bucket in VALID_TTL_BUCKETS:
                    split[f"ephemeral_{bucket}_input_tokens"] = (
                        receipt[1] if bucket == applied else 0
                    )
            changed = True
    else:  # /v1/responses
        kind = obj.get("type")
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            response = obj.get("response")
            if isinstance(response, dict):
                usage = response.setdefault("usage", {})
                if isinstance(usage, dict):
                    _deep_merge_usage(usage, _receipt_to_usage_patch(receipt, route))
                response["prompt_cache_options"] = _applied_options(requested_ttl, receipt[2])
                changed = True
    if not changed:
        return line
    body = json.dumps(obj, ensure_ascii=False)
    prefix = line[: len(line) - len(line.lstrip())]
    return f"{prefix}data: {body}"


async def _stream_relay(upstream: httpx.Response, route: str, requested_ttl: str | None,
                        receipt: tuple[int, int, int] | None):
    """Split the upstream byte stream into complete SSE lines, patch the
    accounting events, and re-emit with identical framing. The upstream is
    closed on normal exhaustion AND on client disconnect (GeneratorExit)."""
    buffer = b""
    state: dict = {}
    try:
        async for chunk in upstream.aiter_bytes():
            if receipt is None:
                yield chunk
                continue
            buffer += chunk
            *complete, buffer = buffer.split(b"\n")
            for line_bytes in complete:
                line = line_bytes.decode("utf-8", "replace")
                yield (_patch_sse_line(line, receipt, route, requested_ttl, state) + "\n").encode("utf-8")
        if buffer:
            line = buffer.decode("utf-8", "replace")
            yield _patch_sse_line(line, receipt, route, requested_ttl, state).encode("utf-8")
    finally:
        await upstream.aclose()


# ---------------------------------------------------------------------------
# Core proxy
# ---------------------------------------------------------------------------

async def _proxy(request: Request, route: str) -> Response:
    body_bytes = await request.body()

    admission: JSONResponse | None = None
    body_obj: dict | None = None
    if request.method == "POST" and body_bytes:
        try:
            body_obj = json.loads(body_bytes)
        except ValueError:
            body_obj = None
        if isinstance(body_obj, dict):
            if route == "/v1/messages":
                admission = _messages_admission(body_obj)
            elif route in ("/v1/responses", "/v1/chat/completions"):
                admission = _options_admission(body_obj)
            if admission is None and route == "/v1/chat/completions":
                admission = _chat_tool_choice_admission(body_obj)
            if admission is not None:
                return admission

    requested_ttl, ttl_seconds = (None, None)
    if isinstance(body_obj, dict):
        requested_ttl, ttl_seconds = _extract_hint(body_obj, route)

    headers = _clean_request_headers(request.headers)
    if ttl_seconds is not None:
        headers[HINT_HEADER] = str(ttl_seconds)

    upstream_req = client.build_request(
        request.method,
        route,
        params=dict(request.query_params),
        headers=headers,
        content=body_bytes,
    )
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.RequestError as exc:
        # Loopback smg down (startup race, crash): every route — including
        # /health and /v1/models probes — must fail, never read as healthy.
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "api_error", "code": "upstream_unavailable",
                               "message": f"Router gateway unreachable: {exc}", "param": None}},
        )
    status = upstream.status_code
    is_sse = "text/event-stream" in (upstream.headers.get("content-type") or "").lower()
    success = 200 <= status < 300
    try:
        receipt = None
        if requested_ttl is not None and success:
            receipt = _parse_receipt(upstream.headers.get(RECEIPT_HEADER))
        resp_headers = _clean_response_headers(upstream.headers)
        if receipt is not None:
            resp_headers.update(_receipt_to_msh(receipt))

        if is_sse and success:
            return StreamingResponse(
                _stream_relay(upstream, route, requested_ttl, receipt),
                status_code=status,
                headers=resp_headers,
                media_type="text/event-stream",
            )
        raw = await upstream.aread()
        if receipt is not None:
            raw = _apply_receipt_json(raw, receipt, route, requested_ttl)
        return Response(raw, status_code=status, headers=resp_headers)
    finally:
        # Streaming bodies close their upstream in _stream_relay (also on
        # client disconnect); every other path closes here.
        if not (is_sse and success):
            await upstream.aclose()


@app.post("/v1/messages")
async def v1_messages(request: Request) -> Response:
    return await _proxy(request, "/v1/messages")


@app.post("/v1/responses")
async def v1_responses(request: Request) -> Response:
    return await _proxy(request, "/v1/responses")


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request) -> Response:
    return await _proxy(request, "/v1/chat/completions")


# Everything else (models, health, generate, metrics, ...) passes through
# untouched on both directions.
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def passthrough(request: Request, path: str) -> Response:
    return await _proxy(request, "/" + path)


def main() -> None:
    port = int(os.environ.get("K3_KVV_FACADE_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
