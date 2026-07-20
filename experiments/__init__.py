"""Leak-safe, artifact-first experiment framework for GraphEval × HalluGraph.

The package deliberately contains orchestration, data lineage and evaluation plumbing;
the scientific detector implementations remain in :mod:`graph_eval` and :mod:`src`.
"""
from .contracts import DetectionInput, DetectionResult, DetectorProtocol

__all__ = ["DetectionInput", "DetectionResult", "DetectorProtocol"]
__version__ = "0.1.0"
