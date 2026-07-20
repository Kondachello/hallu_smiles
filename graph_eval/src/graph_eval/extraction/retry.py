"""Transport retry policy: retry transient faults, fail fast on config/auth.

Error classification is duck-typed so it works with the real ``openai`` exception
types without importing them: a numeric ``status_code`` of 429 or 5xx is transient;
other 4xx are terminal; timeout/connection errors (by class name) are transient.
"""
from __future__ import annotations

import random
import time
from typing import Callable

_RETRYABLE_NAME_HINTS = ("timeout", "connection", "apiconnection")


def classify_error(exc: BaseException) -> str:
    """Return ``"retry"`` for transient transport faults, else ``"fail"``."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return "retry" if (code == 429 or 500 <= code < 600) else "fail"
    name = type(exc).__name__.lower()
    if any(hint in name for hint in _RETRYABLE_NAME_HINTS):
        return "retry"
    return "fail"


class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 5,
        base: float = 2.0,
        *,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] | None = None,
    ):
        self.max_retries = max_retries
        self.base = base
        self._sleep = sleep
        self._jitter = jitter if jitter is not None else (lambda: random.uniform(0.0, 0.25))

    def run(self, fn: Callable[[], object]):
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if classify_error(exc) == "fail" or attempt >= self.max_retries:
                    raise
                self._sleep(self.base ** attempt + self._jitter())
                attempt += 1
