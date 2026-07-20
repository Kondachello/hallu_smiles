"""NLI protocol.  Returns ``p_consistent`` in [0, 1] for (premise, hypothesis)."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class NLIModel(Protocol):
    model_label: str
    revision: str | None

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """``pairs`` are (premise=context, hypothesis=verbalized triple).

        Returns one ``p_consistent`` per pair; higher = better supported.
        """
        ...
