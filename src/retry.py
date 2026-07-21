"""Shared transient-retry timing for gateway-backed components.

This module deliberately contains no provider client code.  It extracts a
standard ``Retry-After`` header from common wrapped HTTP exceptions and combines
it with bounded exponential full-jitter backoff.  The caller still decides
which exception classes are transient.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any


def _exception_chain(exc: BaseException | None) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def retry_after_seconds(exc: BaseException | None) -> float | None:
    """Return a non-negative ``Retry-After`` delay from a wrapped response."""
    for item in _exception_chain(exc):
        for owner in (item, getattr(item, "response", None)):
            headers: Any = getattr(owner, "headers", None)
            if headers is None:
                continue
            try:
                value = headers.get("Retry-After") or headers.get("retry-after")
            except (AttributeError, TypeError):
                continue
            if value is None:
                continue
            try:
                return max(0.0, float(str(value).strip()))
            except ValueError:
                try:
                    target = parsedate_to_datetime(str(value))
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, IndexError, OverflowError):
                    continue
    return None


class WaitRetryAfterOrExponentialJitter:
    """Tenacity wait callable with a bounded ``Retry-After`` override.

    The jitter prevents a synchronized retry burst.  When a gateway gives a
    longer explicit delay, honour it up to the configured maximum pause.
    """

    def __init__(self, base_seconds: float, max_seconds: float):
        self.base_seconds = max(0.0, float(base_seconds))
        self.max_seconds = max(0.0, float(max_seconds))

    def __call__(self, retry_state: Any) -> float:
        attempt = max(1, int(getattr(retry_state, "attempt_number", 1)))
        cap = min(self.max_seconds, self.base_seconds * (2 ** (attempt - 1)))
        jittered = random.uniform(0.0, cap) if cap > 0 else 0.0
        outcome = getattr(retry_state, "outcome", None)
        exception = outcome.exception() if outcome is not None and outcome.failed else None
        retry_after = retry_after_seconds(exception)
        if retry_after is None:
            return jittered
        return min(self.max_seconds, max(jittered, retry_after))
