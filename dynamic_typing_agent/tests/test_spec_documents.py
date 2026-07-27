from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_architecture_documents_exist_and_are_substantial() -> None:
    required = {
        "ARCHITECTURE.md": 3000,
        "IMPLEMENTATION_PLAN.md": 7000,
        "PROMPT_CATALOG.md": 1500,
        "INTEGRATION.md": 1800,
        "TEST_PLAN.md": 1800,
    }
    for name, minimum_length in required.items():
        document = ROOT / "docs" / name
        assert document.is_file()
        assert len(document.read_text(encoding="utf-8")) >= minimum_length


def test_plan_preserves_major_safety_boundaries() -> None:
    plan = (ROOT / "docs" / "IMPLEMENTATION_PLAN.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "do not modify hallugraph matching",
        "no answer field",
        "neutral is not contradiction",
        "cache-only cannot become live",
        "no gold label",
    ):
        assert phrase in plan

