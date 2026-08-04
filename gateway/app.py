"""FastAPI application deployed to Cloud Run.

No request body is logged here. Cloud Run access logs contain route/status only;
the application deliberately emits no prompts, completions, or credentials.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from .core import (
    GatewayError,
    GatewaySettings,
    authenticate,
    gateway_manifest,
    openai_response,
    parse_chat_request,
    translate_vertex_error,
)


# ``/openapi.json`` is an application endpoint too.  Disable it alongside the
# interactive docs so the bearer check protects every route Cloud Run exposes.
app = FastAPI(
    title="HalluGraph Vertex gateway",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@lru_cache(maxsize=1)
def settings() -> GatewaySettings:
    return GatewaySettings.from_env()


@lru_cache(maxsize=1)
def vertex_client() -> Any:
    from google import genai

    cfg = settings()
    return genai.Client(vertexai=True, project=cfg.project, location=cfg.location)


def _authorise(authorization: str | None) -> GatewaySettings:
    cfg = settings()
    authenticate(authorization, cfg.api_key)
    return cfg


@app.exception_handler(GatewayError)
async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.envelope())


@app.get("/healthz")
async def healthz(authorization: str | None = Header(default=None)) -> dict[str, str]:
    cfg = _authorise(authorization)
    return {"status": "ok", "protocol": gateway_manifest(cfg)["protocol"]}


@app.get("/v1/hallu/manifest")
async def manifest(authorization: str | None = Header(default=None)) -> dict[str, str]:
    return gateway_manifest(_authorise(authorization))


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    cfg = _authorise(authorization)
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001 - translate only the public wire failure
        raise GatewayError(400, "request body must contain valid JSON") from exc
    parsed = parse_chat_request(payload, cfg)
    try:
        from google.genai import types

        contents = [
            types.Content(
                role=entry["role"],
                parts=[types.Part.from_text(text=entry["text"])],
            )
            for entry in parsed["contents"]
        ]
        generation: dict[str, Any] = {"temperature": parsed["temperature"]}
        if parsed["system_instruction"] is not None:
            generation["system_instruction"] = parsed["system_instruction"]
        if parsed["max_output_tokens"] is not None:
            generation["max_output_tokens"] = parsed["max_output_tokens"]
        if parsed["response_json_schema"] is not None:
            generation["response_mime_type"] = "application/json"
            generation["response_json_schema"] = parsed["response_json_schema"]
        if parsed["response_logprobs"]:
            generation["response_logprobs"] = True
            # Vertex returns selected-token log probabilities even when no
            # top-k alternatives are requested.  Omitting the field for zero
            # avoids needlessly sending candidate distributions over the wire.
            if parsed["top_logprobs"]:
                generation["logprobs"] = parsed["top_logprobs"]
        response = vertex_client().models.generate_content(
            model=cfg.vertex_model,
            contents=contents,
            config=types.GenerateContentConfig(**generation),
        )
    except GatewayError:
        raise
    except Exception as exc:  # noqa: BLE001 - SDK errors vary between versions
        raise translate_vertex_error(exc) from exc
    return openai_response(response, cfg)
