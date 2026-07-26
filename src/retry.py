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


def transient_http_status(exc: BaseException | None) -> int | None:
    """Return an explicit HTTP status from a wrapped transient exception."""
    for item in _exception_chain(exc):
        response = getattr(item, "response", None)
        status = getattr(item, "status_code", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is None:
            continue
        try:
            return int(status)
        except (TypeError, ValueError):
            continue
    return None


def is_rate_limit_error(exc: BaseException | None) -> bool:
    """Recognise a provider capacity/rate rejection, including wrapped 429s."""
    if transient_http_status(exc) == 429:
        return True
    for item in _exception_chain(exc):
        if type(item).__name__ == "RateLimitError":
            return True
    message = " ".join(str(item) for item in _exception_chain(exc)).lower()
    return "rate limit" in message or "rate_limit" in message


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

    def __init__(
        self,
        base_seconds: float,
        max_seconds: float,
        *,
        rate_limit_cooldown_max_seconds: float | None = None,
    ):
        self.base_seconds = max(0.0, float(base_seconds))
        self.max_seconds = max(0.0, float(max_seconds))
        self.rate_limit_cooldown_max_seconds = max(
            self.max_seconds,
            float(rate_limit_cooldown_max_seconds)
            if rate_limit_cooldown_max_seconds is not None
            else self.max_seconds,
        )

    def __call__(self, retry_state: Any) -> float:
        attempt = max(1, int(getattr(retry_state, "attempt_number", 1)))
        outcome = getattr(retry_state, "outcome", None)
        exception = outcome.exception() if outcome is not None and outcome.failed else None
        cap_limit = (
            self.rate_limit_cooldown_max_seconds
            if is_rate_limit_error(exception)
            else self.max_seconds
        )
        cap = min(cap_limit, self.base_seconds * (2 ** (attempt - 1)))
        jittered = random.uniform(0.0, cap) if cap > 0 else 0.0
        retry_after = retry_after_seconds(exception)
        if retry_after is None:
            return jittered
        if is_rate_limit_error(exception):
            # The gateway's admission-control delay is more informative than
            # a large exponential cap. Add only a small de-synchronisation
            # jitter, rather than idling an extra fifteen minutes after it has
            # explicitly invited a retry in (say) one second.
            cooldown_jitter = random.uniform(0.0, min(5.0, max(1.0, retry_after * 0.1)))
            return min(cap_limit, retry_after + cooldown_jitter)
        return min(cap_limit, max(jittered, retry_after))


class StopAfterAttemptsExceptRateLimit:
    """Bound potentially billable retries while allowing 429 cooldown/recovery.

    A 429 is provider admission control, normally returned before generation.
    It should wait and resume, whereas a timeout can follow an upstream token
    generation that did not return a response and needs a finite budget.
    """

    def __init__(self, max_attempts: int):
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = int(max_attempts)

    def __call__(self, retry_state: Any) -> bool:
        outcome = getattr(retry_state, "outcome", None)
        exception = outcome.exception() if outcome is not None and outcome.failed else None
        if is_rate_limit_error(exception):
            return False
        return int(getattr(retry_state, "attempt_number", 1)) >= self.max_attempts
