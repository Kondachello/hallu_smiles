"""No-gold materialization of the deterministic historical QA selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data import load_instances
from src.sampling import qa_sample_quotas, select_qa_sample

from .ragtruth import materialize_one_response_no_gold


def materialize_historical_qa_no_gold(
    data_dir: str | Path,
    *,
    qa_sample_size: int = 100,
    qa_test_fraction: str = "0.2",
    sample_seed: int = 42,
) -> list[dict[str, Any]]:
    """Recreate the recorded QA selection and project it to detector-safe rows.

    The historical selection algorithm uses its recorded split/label quotas only to
    reproduce the already-existing 100-QA cache population.  Its returned rows never
    contain a gold label, quality flag, or annotation span.
    """
    root = Path(data_dir)
    train_sources, test_sources = qa_sample_quotas(qa_sample_size, qa_test_fraction)
    selected = select_qa_sample(
        load_instances(root),
        seed=sample_seed,
        train_sources=train_sources,
        test_sources=test_sources,
    )
    rows = [
        materialize_one_response_no_gold(
            root / "source_info.jsonl",
            root / "response.jsonl",
            response_id=item.response_id,
        )
        for item in selected
    ]
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["source_id"]), str(row["response_id"])))
