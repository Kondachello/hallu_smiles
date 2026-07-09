#!/usr/bin/env python3
"""Generate a tiny synthetic RAGTruth-format dataset for offline plumbing/smoke tests.

NOT real data -- only exercises the pipeline end-to-end without an API key.
    python tests/make_fixture.py <out_dir>
"""
import json
import sys
from pathlib import Path

TASKS = ["QA", "Data2txt", "Summary"]


def make(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sources, responses = [], []
    sid = 1000
    rid = 5000
    for t_i, task in enumerate(TASKS):
        for k in range(4):  # 4 sources per task
            sid += 1
            split = "train" if k < 2 else "test"
            if task == "QA":
                src_info = {
                    "question": f"where is riverton university located in year {k}",
                    "passages": (
                        f"passage 1: Riverton University is located in Fairhaven County. "
                        f"It was founded in nineteen fifty. The chancellor is Diana Merrick. "
                        f"passage 2: Fairhaven County borders the Silverlake district."
                    ),
                }
            elif task == "Data2txt":
                src_info = {
                    "name": f"Deja Vu Cafe {k}",
                    "city": "Fairhaven",
                    "categories": "Restaurants, Coffee",
                    "attributes": {"RestaurantsReservations": False, "Music": None},
                    "business_stars": 3.0,
                    "review_info": [
                        {"review_stars": 4.0, "review_text": "Pleasant coffee and friendly baristas."},
                    ],
                }
            else:  # Summary
                src_info = {
                    "question": "", "passages": "",
                }
                src_info = (
                    "Fairhaven County council approved a new transit plan on Monday. "
                    "Chancellor Diana Merrick praised the Silverlake extension. "
                    "The project will cost forty million dollars over five years."
                )
            sources.append({
                "source_id": str(sid), "task_type": task, "source": "SYNTH",
                "source_info": src_info, "prompt": "synthetic prompt",
            })

            # faithful response (y=0): reuses context entities
            rid += 1
            responses.append({
                "id": str(rid), "source_id": str(sid), "model": "gpt-4-synth",
                "temperature": 0.0,
                "response": ("Riverton University is located in Fairhaven County and was "
                             "founded in nineteen fifty; the chancellor is Diana Merrick.")
                if task == "QA" else
                ("Deja Vu Cafe is a coffee restaurant in Fairhaven with pleasant coffee "
                 "and friendly baristas.") if task == "Data2txt" else
                ("Fairhaven County council approved a transit plan; Chancellor Diana "
                 "Merrick praised the Silverlake extension costing forty million dollars."),
                "labels": [], "split": split, "quality": "good",
            })

            # hallucinated response (y=1): introduces unsupported entities
            rid += 1
            responses.append({
                "id": str(rid), "source_id": str(sid), "model": "llama-2-7b-synth",
                "temperature": 0.9,
                "response": ("Riverton University is located in Brightmoor Province, founded "
                             "in eighteen eighty by Emperor Castellan near Mount Verwood.")
                if task == "QA" else
                ("Deja Vu Cafe is a luxury steakhouse in Brightmoor famous for lobster "
                 "and live jazz orchestras every midnight.") if task == "Data2txt" else
                ("The Brightmoor senate rejected the plan; Emperor Castellan condemned the "
                 "Verwood tunnel costing nine billion doubloons."),
                "labels": [{"start": 0, "end": 10, "text": "hallucinated",
                            "label_type": "Evident Baseless Info"}],
                "split": split, "quality": "good",
            })

    (out_dir / "source_info.jsonl").write_text(
        "\n".join(json.dumps(s) for s in sources) + "\n", encoding="utf-8")
    (out_dir / "response.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses) + "\n", encoding="utf-8")
    print(f"wrote {len(sources)} sources, {len(responses)} responses to {out_dir}")


if __name__ == "__main__":
    make(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixture_data"))
