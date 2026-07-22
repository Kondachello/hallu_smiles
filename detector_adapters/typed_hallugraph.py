"""Contract-only typed HalluGraph variant over an immutable shared graph bundle.

The adapter is intentionally score-preserving: the current ``UnknownTypeProvider``
annotates every node as unknown, then delegates unchanged scoring to the baseline
HalluGraph adapter.  This proves the three-way experiment infrastructure before a
future model-backed type agent changes matching or metrics.  Its contract is in
``docs/dynamic-typing-experiment-infrastructure.md``.
"""
from __future__ import annotations

from typing import Any, Mapping

from graph_eval.types import DetectionInput, DetectionResult, STATUS_FAILED
from src.cache import CacheOnlyMissError


class TypedHalluGraphAdapter:
    """Wrap an untyped HalluGraph adapter with a no-gold dynamic-typing provider."""

    method_name = "hallugraph"
    variant_name = "dynamic_types_stub_v1"
    variant_id = "hallugraph_dynamic_types_stub_v1"

    def __init__(
        self,
        baseline: Any,
        *,
        typing_provider: Any,
        shared_graph_provider: Any,
        variant_name: str | None = None,
        variant_id: str | None = None,
    ):
        self.baseline = baseline
        self.typing_provider = typing_provider
        self.shared_graph_provider = shared_graph_provider
        self.method_name = str(getattr(baseline, "method_name", self.method_name))
        self.variant_name = variant_name or self.variant_name
        self.variant_id = variant_id or self.variant_id

    def predict(self, item: DetectionInput) -> DetectionResult:
        try:
            bundle = self.shared_graph_provider.prepare_instance(item)
            annotations = self.typing_provider.prepare(item=item, bundle=bundle)
        except CacheOnlyMissError:
            raise
        except Exception as exc:  # type-provider errors must remain explicit states
            return DetectionResult(
                item.response_id,
                item.source_id,
                self.method_name,
                None,
                {},
                (),
                STATUS_FAILED,
                {"stage": "dynamic_typing", "error": repr(exc)},
                {},
                {},
            )
        baseline_result = self.baseline.predict(item)
        components = {
            **dict(baseline_result.components),
            "dynamic_typing": annotations.summary(),
            "dynamic_typing_mode": "contract_stub_score_preserving",
        }
        refs = {
            **dict(baseline_result.artifact_refs),
            **bundle.reference(),
            **annotations.reference(),
        }
        return DetectionResult(
            baseline_result.response_id,
            baseline_result.source_id,
            baseline_result.method,
            baseline_result.raw_score,
            components,
            baseline_result.flagged_unit_ids,
            baseline_result.status,
            baseline_result.failure,
            baseline_result.usage,
            refs,
        )

    def export_artifact_records(self) -> Mapping[str, Any]:
        return self.typing_provider.artifact_records()


__all__ = ["TypedHalluGraphAdapter"]
