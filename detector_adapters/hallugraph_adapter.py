"""HalluGraph exposed through graph_eval's DetectionInput/DetectionResult contract.

The adapter is a thin wrapper: it calls the SAME primitives the offline pipeline
uses -- ``run.build_refgraph`` and ``src.metrics.score_response`` -- and then reads
``EG`` / ``RP`` / ``cfi_for_mode`` / ``h_for_mode`` exactly as ``run.build_rows``
does.  Parity therefore holds by construction; ``run.py`` is not modified and stays
the regression oracle.  ``raw_score`` = H = 1 - CFI (higher = more hallucination);
EG/RP are also returned so the experiment framework can recompute CFI at its own
tuned alpha.
"""
from __future__ import annotations

import run
from graph_eval.types import (
    STATUS_EMPTY_GRAPH,
    STATUS_FAILED,
    STATUS_OK,
    DetectionInput,
    DetectionResult,
)
from src.cache import CacheOnlyMissError
from src.metrics import score_response

METHOD = "hallugraph"
_VERIFIED_MODES = {"support", "support-critical"}


class HalluGraphAdapter:
    def __init__(
        self,
        cfg,
        extractor,
        embedder,
        *,
        alpha: float,
        tau_e: float | None = None,
        tau_r: float | None = None,
        relation_mode: str = "strict",
        verifier=None,
        verifier_matching_params: dict | None = None,
        critical_pipeline=None,
        method: str = METHOD,
    ):
        self.cfg = cfg
        self.extractor = extractor
        self.embedder = embedder
        self.alpha = float(alpha)
        self.tau_e = tau_e
        self.tau_r = tau_r
        self.relation_mode = relation_mode
        self.verifier = verifier
        self.verifier_matching_params = verifier_matching_params
        self.critical_pipeline = critical_pipeline
        self.method = method

    def predict(self, item: DetectionInput) -> DetectionResult:
        mode = self.relation_mode
        try:
            gc, gq = self.extractor.extract_reference(item.context, item.query)
            ga = self.extractor.extract(item.response, kind="response")
        except CacheOnlyMissError:
            raise
        except Exception as exc:  # noqa: BLE001 - extraction failure is a state
            return self._failed(item, {"stage": "extraction", "error": repr(exc)})

        refgraph = run.build_refgraph(self.cfg, gc, gq, self.embedder, self.tau_e, self.tau_r)
        try:
            result = score_response(
                ga, refgraph, gc, gq,
                context=item.context, query=item.query,
                verifier=self.verifier if mode in _VERIFIED_MODES else None,
                verifier_matching_params=self.verifier_matching_params,
                answer_text=item.response if mode == "support-critical" else None,
                critical_pipeline=self.critical_pipeline if mode == "support-critical" else None,
            )
        except CacheOnlyMissError:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._failed(item, {"stage": "score", "error": repr(exc)})

        h = result.h_for_mode(self.alpha, mode)
        cfi = result.cfi_for_mode(self.alpha, mode)
        components = {
            "EG": result.EG,
            "RP": result.RP,
            "RP_strict": result.RP_strict,
            "RP_support": result.RP_support,
            "RP_support_defined": result.RP_support_defined,
            "CFI": cfi,
            "alpha": self.alpha,
            "relation_mode": mode,
            "tau_e": self.tau_e,
            "tau_r": self.tau_r,
            "Vc": result.Vc, "Ec": result.Ec,
            "Vq": result.Vq, "Eq": result.Eq,
            "Va": result.Va, "Ea": result.Ea,
            "unscorable": result.unscorable,
            "ref_empty": result.ref_empty,
            "H_eg": result.eg_only_h(),
            "ungrounded_entities": list(result.ungrounded_entities),
            "unsupported_relations": [list(r) for r in result.unsupported_relations],
        }

        if result.unscorable or h is None:
            # Va == 0 (no answer entities) => EG undefined => distinct state, no score.
            return DetectionResult(
                item.response_id, item.source_id, self.method, None,
                {**components, "reason": "unscorable_or_empty"},
                (), STATUS_EMPTY_GRAPH, None, {"extractor_calls": 2}, {},
            )

        flagged = tuple(
            [f"entity:{name}" for name in result.ungrounded_entities]
            + [f"relation:{'|'.join(map(str, edge))}" for edge in result.unsupported_relations]
        )
        return DetectionResult(
            item.response_id, item.source_id, self.method,
            float(h), components, flagged, STATUS_OK, None, {"extractor_calls": 2}, {},
        )

    def _failed(self, item: DetectionInput, failure: dict) -> DetectionResult:
        return DetectionResult(
            item.response_id, item.source_id, self.method, None,
            {}, (), STATUS_FAILED, failure, {}, {},
        )
