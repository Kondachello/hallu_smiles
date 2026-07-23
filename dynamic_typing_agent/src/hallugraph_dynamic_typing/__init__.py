"""Standalone dynamic entity typing agent contracts.

The graph runtime is intentionally not implemented in the architecture workstream.
"""

from .graph_spec import MODEL_NODE_IDS, NODE_SPECS, NodeKind, NodeSpec
from .types import DecisionAction, EvidenceLevel, NliVerdict, RunStatus

__all__ = [
    "DecisionAction",
    "DynamicTypingAgent",
    "EvidenceLevel",
    "MODEL_NODE_IDS",
    "NODE_SPECS",
    "NliVerdict",
    "NodeKind",
    "NodeSpec",
    "RunStatus",
]

__version__ = "0.2.0"


def __getattr__(name: str):
    """Avoid importing LangGraph when a local artifact-only utility is used."""
    if name == "DynamicTypingAgent":
        from .agent import DynamicTypingAgent

        return DynamicTypingAgent
    raise AttributeError(name)
