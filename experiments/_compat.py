"""Development-tree import compatibility.

GraphEval intentionally has its own installable ``src`` layout.  During repository-root
development (including ``python -m experiments``) it has not necessarily been installed,
so make that source tree importable without affecting an installed environment.
"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_graph_eval_importable() -> None:
    source_root = Path(__file__).resolve().parents[1] / "graph_eval" / "src"
    text = str(source_root)
    if source_root.is_dir() and text not in sys.path:
        sys.path.insert(0, text)
