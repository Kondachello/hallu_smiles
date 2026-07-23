"""Dependency-inversion ports for model, NLI, cache and artifact implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class StructuredModelPort(Protocol):
    def invoke(
        self,
        *,
        operation: str,
        messages: tuple[Mapping[str, str], ...],
        output_schema: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...


class NliPort(Protocol):
    def verify(
        self,
        *,
        hypothesis_kind: str,
        premise: str,
        hypothesis: str,
        evidence_span_ids: tuple[str, ...],
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...


class CachePort(Protocol):
    def get(self, namespace: str, key: str) -> Mapping[str, Any] | None: ...

    def put_immutable(self, namespace: str, key: str, value: Mapping[str, Any]) -> None: ...


class ArtifactSinkPort(Protocol):
    def append(self, relative_path: str, record: Mapping[str, Any]) -> None: ...

