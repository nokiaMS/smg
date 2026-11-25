"""Behavioral tests for the K3 KVV facade's Chat billing group.

Covers AGDC2-79: on /v1/chat/completions the facade publishes
``Msh-Usage-Prompt-Tokens`` (P) and ``Msh-Usage-Cached-Tokens`` (C) from the
worker's private receipt C,W,A plus the separately reported public basis P
(``x-sglang-reported-prompt-tokens``), never computing P from C+W. Messages
and Responses keep their legacy contract (no P pair).

The upstream SMG gateway is faked with a local ASGI app so no real router,
GPU or worker is needed.
"""

from __future__ import annotations

import json

import httpx
import pytest
import smg.k3_kvv_facade as facade
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

HINT = "x-sglang-cache-retention-ttl"
RECEIPT = "x-sglang-prompt-cache-receipt"
REPORTED = "x-sglang-reported-prompt-tokens"


# ---------------------------------------------------------------------------
# Fake upstream (stands in for the loopback SMG Rust gateway)
# ---------------------------------------------------------------------------


def make_upstream(
    extra_headers: dict | None = None,
    body: dict | None = None,
    sse_frames: list[dict] | None = None,
):
    app = FastAPI()
    seen: list[httpx.Headers] = []

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def catch_all(request: Request, path: str):
        seen.append(request.headers)
        if sse_frames is not None:
            lines = []
            for frame in sse_frames:
                lines.append("data: " + json.dumps(frame))
            lines.append("data: [DONE]")
            return StreamingResponse(
                iter([("\n".join(lines) + "\n").encode()]),
                status_code=200,
                headers={**(extra_headers or {}), "content-type": "text/event-stream"},
            )
        return Response(
            json.dumps(body or {}).encode(),
            status_code=200,
            headers={"content-type": "application/json", **(extra_headers or {})},
        )

    return app, seen


async def call_facade(
    upstream_app, method: str, path: str, json_body=None, headers: dict | None = None
):
    """Send a request through the facade's own FastAPI app with its shared
    upstream AsyncClient rewired to the fake upstream."""
    _wire(upstream_app)
    transport = httpx.ASGITransport(app=facade.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://facade-test") as c:
        return await c.request(method, path, json=json_body, headers=headers)


@pytest.fixture(autouse=True)
def reset_client():
    """Point the facade's shared AsyncClient at the fake upstream for each
    test and restore the original loopback wiring afterwards."""
    saved = (facade.client._transport, facade.client._base_url)
    yield
    facade.client._transport, facade.client._base_url = saved


def _wire(upstream_app):
    facade.client._transport = httpx.ASGITransport(app=upstream_app)
    facade.client._base_url = httpx.URL("http://upstream-test")


CHAT_OPT_IN = {
    "model": "kimi-k3",
    "prompt_cache_options": {"mode": "implicit", "ttl": "5m"},
    "messages": [{"role": "user", "content": "hi"}],
}


# ---------------------------------------------------------------------------
# Header emission: complete group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_complete_group_headers(reset_client):
    upstream, seen = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        body={
            "id": "1",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 105,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 70},
            },
        },
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert resp.status_code == 200
    assert resp.headers["msh-usage-prompt-tokens"] == "100"
    assert resp.headers["msh-usage-cached-tokens"] == "70"
    assert resp.headers["msh-usage-cache-read-tokens"] == "70"
    assert resp.headers["msh-usage-cache-write-tokens-5m"] == "30"
    assert resp.headers["msh-usage-cache-write-tokens-1h"] == "0"
    # Private markers never escape the client-facing iface.
    assert REPORTED not in resp.headers
    assert RECEIPT not in resp.headers
    # Body agrees with measured P.
    assert resp.json()["usage"]["prompt_tokens"] == 100
    assert resp.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 70
    assert resp.json()["usage"]["prompt_tokens_details"]["cache_write_tokens"] == 30


@pytest.mark.asyncio
async def test_chat_zero_write_measured_zero_emitted(reset_client):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "50,0,300", REPORTED: "50"},
        body={"choices": [], "usage": {"prompt_tokens": 50}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert resp.headers["msh-usage-prompt-tokens"] == "50"
    assert resp.headers["msh-usage-cached-tokens"] == "50"
    assert resp.headers["msh-usage-cache-write-tokens-5m"] == "0"


@pytest.mark.asyncio
async def test_chat_sse_usage_frame_carries_group(reset_client):
    frames = [
        {"id": "1", "choices": [{"delta": {"content": "he"}}]},
        {
            "id": "1",
            "choices": [],
            "usage": {
                "prompt_tokens": 999,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 70},
            },
        },
    ]
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        sse_frames=frames,
    )
    _wire(upstream)
    resp = await call_facade(
        upstream, "POST", "/v1/chat/completions", {**CHAT_OPT_IN, "stream": True}
    )
    assert resp.status_code == 200
    usage_frame = None
    for line in resp.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            obj = json.loads(line[6:])
            if isinstance(obj.get("usage"), dict):
                usage_frame = obj
    assert usage_frame is not None
    assert usage_frame["usage"]["prompt_tokens"] == 100
    assert usage_frame["usage"]["prompt_tokens_details"]["cached_tokens"] == 70
    # The content frames are untouched.
    assert resp.headers["msh-usage-prompt-tokens"] == "100"
    assert resp.headers["msh-usage-cached-tokens"] == "70"


# ---------------------------------------------------------------------------
# Mixed-version workers: incomplete groups stay incomplete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_legacy_worker_receipt_only_keeps_old_behavior(reset_client):
    """Old worker: valid C,W,A, no P. Legacy headers stay; no partial group."""
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "70,30,300"},
        body={"choices": [], "usage": {"prompt_tokens": 105}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cached-tokens" not in resp.headers
    assert resp.headers["msh-usage-cache-read-tokens"] == "70"
    assert resp.headers["msh-usage-cache-write-tokens-5m"] == "30"
    # Body keeps the worker's own accounting.
    assert resp.json()["usage"]["prompt_tokens"] == 105
    assert resp.json()["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 70,
        "cache_write_tokens": 30,
        "cache_write_ttl_s": 300,
    }


@pytest.mark.asyncio
async def test_chat_reported_only_emits_nothing(reset_client):
    """P without a receipt is not a group; no headers, body untouched."""
    upstream, _ = make_upstream(
        extra_headers={REPORTED: "100"},
        body={"choices": [], "usage": {"prompt_tokens": 105}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cached-tokens" not in resp.headers
    assert "msh-usage-cache-read-tokens" not in resp.headers
    assert "msh-usage-cache-write-tokens-5m" not in resp.headers
    assert resp.json()["usage"]["prompt_tokens"] == 105


@pytest.mark.asyncio
async def test_chat_missing_receipt_and_reported_plain_passthrough(reset_client):
    upstream, _ = make_upstream(body={"choices": [], "usage": {"prompt_tokens": 10}})
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert "msh-usage-cache-read-tokens" not in resp.headers
    assert "msh-usage-prompt-tokens" not in resp.headers


@pytest.mark.asyncio
async def test_chat_contradictory_c_plus_w_exceeds_p_suppresses_group(reset_client):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "90"},  # C+W=100 > P=90
        body={"choices": [], "usage": {"prompt_tokens": 90}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cached-tokens" not in resp.headers
    # The whole evidence channel is untrusted, not just the pair.
    assert "msh-usage-cache-read-tokens" not in resp.headers
    assert "msh-usage-cache-write-tokens-5m" not in resp.headers
    assert resp.json()["usage"]["prompt_tokens"] == 90


# ---------------------------------------------------------------------------
# Invalid / duplicate / adversarial evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_invalid_receipt_keeps_valid_reported_silent(reset_client):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "not-a-receipt", REPORTED: "100"},
        body={"choices": [], "usage": {"prompt_tokens": 100}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cache-read-tokens" not in resp.headers
    assert resp.json()["usage"]["prompt_tokens"] == 100


@pytest.mark.asyncio
async def test_chat_negative_or_overflow_reported_rejected(reset_client):
    for bad in ("-1", "2147483648", "1e5", "  ", "abc", "1_00", "+100"):
        upstream, _ = make_upstream(
            extra_headers={RECEIPT: "70,30,300", REPORTED: bad},
            body={"choices": [], "usage": {"prompt_tokens": 100}},
        )
        _wire(upstream)
        resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
        # Receipt stays valid on its own; the P pair never appears.
        assert resp.headers.get("msh-usage-cache-read-tokens") == "70"
        assert "msh-usage-prompt-tokens" not in resp.headers
        assert "msh-usage-cached-tokens" not in resp.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["20,0,0", "2_0,0,300", "20,+0,300", "0,2147483648,300"])
async def test_unprotected_or_invalid_receipt_has_no_billing_pair(wire):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: wire, REPORTED: "100"},
        body={"choices": [], "usage": {"prompt_tokens": 100}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cached-tokens" not in resp.headers


@pytest.mark.asyncio
async def test_chat_edge_values_accepted(reset_client):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "0,0,300", REPORTED: "2147483647"},
        body={"choices": [], "usage": {"prompt_tokens": 1}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert resp.headers["msh-usage-prompt-tokens"] == "2147483647"
    assert resp.headers["msh-usage-cached-tokens"] == "0"


@pytest.mark.asyncio
async def test_chat_upstream_msh_headers_are_stripped(reset_client):
    """A worker (or a reflection bug) minting public Msh headers directly
    must not leak duplicates next to the facade's own evidence."""
    upstream, _ = make_upstream(
        extra_headers={
            RECEIPT: "70,30,300",
            REPORTED: "100",
            "msh-usage-prompt-tokens": "999",  # contradictory forgery
            "msh-usage-cached-tokens": "999",
            "msh-usage-cache-read-tokens": "123",  # duplicated legacy read
        },
        body={"choices": [], "usage": {"prompt_tokens": 100}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert resp.headers["msh-usage-prompt-tokens"] == "100"
    assert resp.headers["msh-usage-cached-tokens"] == "70"
    assert resp.headers["msh-usage-cache-read-tokens"] == "70"


# ---------------------------------------------------------------------------
# Private-header hygiene on the client-facing side
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_supplied_private_and_public_markers_stripped(reset_client):
    upstream, seen = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        body={"choices": [], "usage": {"prompt_tokens": 100}},
    )
    _wire(upstream)
    resp = await call_facade(
        upstream,
        "POST",
        "/v1/chat/completions",
        CHAT_OPT_IN,
        headers={
            HINT: "9999",  # client cannot mint the hint
            REPORTED: "1",  # nor the reported basis
            RECEIPT: "1,2,3",
            "msh-usage-prompt-tokens": "5",  # nor public evidence
        },
    )
    forwarded = seen[0]
    assert forwarded.get(HINT) == "300"  # facade-minted from the opt-in body
    assert forwarded.get(REPORTED) is None
    assert forwarded.get(RECEIPT) is None
    assert forwarded.get("msh-usage-prompt-tokens") is None
    assert resp.headers["msh-usage-prompt-tokens"] == "100"


@pytest.mark.asyncio
async def test_non_opt_in_chat_gets_no_headers_and_no_hint(reset_client):
    upstream, seen = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        body={"choices": [], "usage": {"prompt_tokens": 100}},
    )
    _wire(upstream)
    resp = await call_facade(
        upstream,
        "POST",
        "/v1/chat/completions",
        {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert forwarded_is_clean(seen)
    for name in (
        "msh-usage-prompt-tokens",
        "msh-usage-cached-tokens",
        "msh-usage-cache-read-tokens",
    ):
        assert name not in resp.headers


def forwarded_is_clean(seen) -> bool:
    h = seen[0]
    return h.get(HINT) is None and h.get(RECEIPT) is None and h.get(REPORTED) is None


# ---------------------------------------------------------------------------
# Responses / Messages keep their legacy contract (no P pair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_route_never_emits_prompt_pair(reset_client):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        body={"object": "response", "usage": {"input_tokens": 100}},
    )
    _wire(upstream)
    resp = await call_facade(
        upstream,
        "POST",
        "/v1/responses",
        {"model": "kimi-k3", "input": "hi", "prompt_cache_options": {"mode": "implicit"}},
    )
    assert resp.status_code == 200
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cached-tokens" not in resp.headers
    assert resp.headers["msh-usage-cache-read-tokens"] == "70"
    assert resp.headers["msh-usage-cache-write-tokens-5m"] == "30"
    usage = resp.json()["usage"]
    assert usage["input_tokens_details"]["cached_tokens"] == 70
    assert usage["input_tokens_details"]["cache_write_tokens"] == 30
    assert "prompt_tokens" not in usage
    assert REPORTED not in resp.headers


@pytest.mark.asyncio
async def test_messages_route_never_emits_prompt_pair(reset_client):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        body={"content": [], "usage": {"input_tokens": 100, "output_tokens": 5}},
    )
    _wire(upstream)
    resp = await call_facade(
        upstream,
        "POST",
        "/v1/messages",
        {
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hi"}],
            "cache_control": {"type": "ephemeral"},
        },
    )
    assert resp.status_code == 200
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cached-tokens" not in resp.headers
    assert resp.headers["msh-usage-cache-read-tokens"] == "70"
    usage = resp.json()["usage"]
    assert usage["cache_read_input_tokens"] == 70
    assert usage["cache_creation_input_tokens"] == 30
    assert usage["cache_creation"]["ephemeral_5m_input_tokens"] == 30
    assert usage["cache_creation"]["ephemeral_1h_input_tokens"] == 0
    assert REPORTED not in resp.headers


# ---------------------------------------------------------------------------
# No timing/usage regression: non-usage frames and bodies untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_no_usage_in_body_only_details_added(reset_client):
    """Body without usage: facade inserts the details (as before) but P
    alignment only applies where usage exists; no usage keys invented beyond
    the legacy behavior."""
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        body={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert resp.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 70


@pytest.mark.asyncio
async def test_chat_error_body_untouched(reset_client):
    upstream, _ = make_upstream(
        extra_headers={RECEIPT: "70,30,300", REPORTED: "100"},
        body={"error": {"message": "boom"}},
    )
    _wire(upstream)
    resp = await call_facade(upstream, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert resp.json() == {"error": {"message": "boom"}}


@pytest.mark.asyncio
async def test_chat_duplicate_receipt_or_reported_rejected(reset_client):
    """The worker middleware mints each private header at most once; a
    duplicated value is a protocol violation, not a tie to break."""

    # httpx Headers support multi-value construction; simulate a response
    # carrying the receipt twice and the reported count twice via raw headers.
    async def duplicate_upstream(request: Request) -> Response:
        resp = Response(
            json.dumps({"choices": [], "usage": {"prompt_tokens": 100}}).encode(),
            status_code=200,
            headers={"content-type": "application/json"},
        )
        resp.raw_headers += [
            (RECEIPT.encode(), b"70,30,300"),
            (RECEIPT.encode(), b"70,30,300"),
            (REPORTED.encode(), b"100"),
            (REPORTED.encode(), b"100"),
        ]
        return resp

    app = FastAPI()
    app.router.add_api_route("/v1/chat/completions", duplicate_upstream, methods=["POST"])
    _wire(app)
    resp = await call_facade(app, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert "msh-usage-prompt-tokens" not in resp.headers
    assert "msh-usage-cached-tokens" not in resp.headers
    # Duplicates void the evidence entirely, legacy half included.
    assert "msh-usage-cache-read-tokens" not in resp.headers
    assert "msh-usage-cache-write-tokens-5m" not in resp.headers
    assert resp.json()["usage"]["prompt_tokens"] == 100


@pytest.mark.asyncio
async def test_chat_upstream_error_uses_no_evidence(reset_client):
    """The receipt is only projected from a successful metered decode; an
    upstream error must not mint Msh headers or rewrite the body."""
    app = FastAPI()

    async def error_upstream(request: Request) -> Response:
        return Response(
            json.dumps({"error": {"message": "upstream boom"}}).encode(),
            status_code=500,
            headers={"content-type": "application/json", RECEIPT: "70,30,300", REPORTED: "100"},
        )

    app.router.add_api_route("/v1/chat/completions", error_upstream, methods=["POST"])
    _wire(app)
    resp = await call_facade(app, "POST", "/v1/chat/completions", CHAT_OPT_IN)
    assert resp.status_code == 500
    assert resp.json() == {"error": {"message": "upstream boom"}}
    for name in (
        "msh-usage-prompt-tokens",
        "msh-usage-cached-tokens",
        "msh-usage-cache-read-tokens",
    ):
        assert name not in resp.headers
