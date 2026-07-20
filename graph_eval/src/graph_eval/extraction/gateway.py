"""Cloud Run / OpenAI-compatible gateway extractor.

The ``openai`` client is imported lazily and can be injected for offline tests.
Bounded repair handles truncation (``finish_reason == 'length'``) and unparseable
output; after ``max_repairs`` a controlled :class:`ExtractionError` is raised (which
the detector records as a ``failed`` state — never a hallucination score). The
gateway secret is read from the environment by name only; it is never logged.
"""
from __future__ import annotations

import os

from ..parser import STATUS_MALFORMED, parse_triples
from .base import ExtractionError, ExtractionOutput
from .prompt import (
    STRUCTURED_RESPONSE_FORMAT,
    build_messages,
    repair_messages,
)
from .retry import RetryPolicy


class GatewayExtractor:
    def __init__(self, config, *, client=None, manifest_sha256: str | None = None, retry=None):
        self.config = config
        self.prompt_profile = config.prompt_profile
        self.manifest_sha256 = manifest_sha256
        self._client = client
        self._retry = retry or RetryPolicy(max_retries=config.max_retries)

    def _ensure_client(self):
        if self._client is None:
            self._client = self._default_client()
        return self._client

    def _default_client(self):
        from openai import OpenAI  # lazy, optional dependency

        base_url = os.environ[self.config.api_base_env].rstrip("/")
        return OpenAI(base_url=base_url, api_key=os.environ[self.config.api_key_env])

    def _call_kwargs(self) -> dict:
        kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.output_mode == "structured_json":
            kwargs["response_format"] = STRUCTURED_RESPONSE_FORMAT
        return kwargs

    def _one_call(self, client, messages, call_kwargs):
        resp = client.chat.completions.create(messages=messages, **call_kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)
        usage = _extract_usage(resp)
        return content, finish, usage

    def extract(self, response_text: str) -> ExtractionOutput:
        base = build_messages(response_text, self.config.output_mode)
        call_kwargs = self._call_kwargs()
        client = self._ensure_client()
        messages = base
        repairs = 0
        while True:
            content, finish, usage = self._retry.run(
                lambda m=messages: self._one_call(client, m, call_kwargs)
            )
            parseable = parse_triples(content).status != STATUS_MALFORMED
            if finish != "length" and parseable:
                usage["extractor_calls"] = 1
                usage["repair_attempts"] = repairs
                return ExtractionOutput(raw_output=content, usage=usage)
            if repairs >= self.config.max_repairs:
                raise ExtractionError(
                    f"unrepairable extractor output (finish={finish}, parseable={parseable})",
                    raw=content,
                    finish=finish,
                )
            messages = repair_messages(base, content)
            repairs += 1


def _extract_usage(resp) -> dict:
    usage = getattr(resp, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "model_fingerprint": getattr(resp, "system_fingerprint", None) or getattr(resp, "model", None),
    }
