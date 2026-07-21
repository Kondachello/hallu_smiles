"""Local HHEM-2.1-Open NLI adapter.

HHEM's ``model.predict([(premise, hypothesis), ...])`` returns a **consistency**
probability in [0, 1] (higher = the hypothesis is better supported by the premise),
which is exactly our ``p_consistent`` — the hallucination flip happens later in
``scoring.p_unsupported``.  ``torch``/``transformers`` are imported lazily so this
module (and its tests, via an injected fake model) load without those deps.
"""
from __future__ import annotations

from typing import Sequence

from ..config import NLIConfig, _UNPINNED_REVISIONS


class HHEMNLIModel:
    def __init__(self, config: NLIConfig, *, model=None, loader=None):
        self.config = config
        self.model_label = config.model_label
        self.revision = config.revision
        self.batch_size = max(1, int(config.batch_size))
        self._model = model      # inject a fake in tests to avoid torch
        self._loader = loader

    def _ensure_model(self):
        if self._model is None:
            self._model = self._loader() if self._loader is not None else self._default_load()
        return self._model

    def _default_load(self):
        if not self.revision or self.revision in _UNPINNED_REVISIONS:
            raise ValueError(
                "HHEM revision must be an exact Hugging Face commit SHA before loading"
            )
        from transformers import AutoModelForSequenceClassification  # lazy, heavy

        return AutoModelForSequenceClassification.from_pretrained(
            self.config.model,
            trust_remote_code=True,
            revision=self.revision,
            local_files_only=True,
        )

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._ensure_model()
        scores: list[float] = []
        pairs = list(pairs)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            scores.extend(_as_floats(model.predict(batch)))
        return scores


def _as_floats(scores) -> list[float]:
    """Coerce a torch tensor / numpy array / list of scores to plain floats."""
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [float(s) for s in scores]
