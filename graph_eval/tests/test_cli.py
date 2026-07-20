import json

import yaml

from graph_eval.cli import main


def _write_config(tmp_path):
    cfg = {"extractor": {"backend": "fake"}, "nli": {"backend": "fake"}, "cache_dir": str(tmp_path / "cache")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _write_instances(tmp_path, rows):
    path = tmp_path / "in.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return str(path)


def _read_predictions(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_predict_writes_predictions(tmp_path, capsys):
    config = _write_config(tmp_path)
    inp = _write_instances(
        tmp_path,
        [
            {"response_id": "r1", "source_id": "s1",
             "context": "Paris is the capital of France.",
             "response": "Paris is the capital of France.", "query": "capital?"},
            {"response_id": "r2", "source_id": "s2",
             "context": "The sky is blue today.", "response": "hi"},  # -> empty graph
        ],
    )
    out = str(tmp_path / "out.jsonl")
    rc = main(["predict", "--config", config, "--input", inp, "--output", out])
    assert rc == 0

    preds = _read_predictions(out)
    assert [p["response_id"] for p in preds] == ["r1", "r2"]
    assert preds[0]["status"] == "ok" and preds[0]["method"] == "grapheval"
    assert preds[1]["status"] == "empty_graph" and preds[1]["raw_score"] is None
    summary = json.loads(capsys.readouterr().out)
    assert summary["written"] == 2


def test_resume_skips_already_done(tmp_path):
    config = _write_config(tmp_path)
    rows = [
        {"response_id": f"r{i}", "source_id": f"s{i}",
         "context": "Paris is the capital of France.",
         "response": "Paris is the capital of France."}
        for i in range(3)
    ]
    inp = _write_instances(tmp_path, rows)
    out = str(tmp_path / "out.jsonl")

    main(["predict", "--config", config, "--input", inp, "--output", out, "--limit", "1"])
    assert len(_read_predictions(out)) == 1

    main(["predict", "--config", config, "--input", inp, "--output", out, "--resume"])
    preds = _read_predictions(out)
    assert sorted(p["response_id"] for p in preds) == ["r0", "r1", "r2"]  # no duplicates


def test_gold_fields_in_input_are_ignored(tmp_path):
    config = _write_config(tmp_path)
    inp = _write_instances(
        tmp_path,
        [{"response_id": "r1", "source_id": "s1",
          "context": "Paris is the capital of France.",
          "response": "Paris is the capital of France.",
          "gold_response_label": "hallucinated",  # must be ignored
          "gold_spans": [{"start": 0, "end": 5}]}],
    )
    out = str(tmp_path / "out.jsonl")
    main(["predict", "--config", config, "--input", inp, "--output", out])
    pred = _read_predictions(out)[0]
    blob = json.dumps(pred)
    assert "gold_response_label" not in blob and "gold_spans" not in blob
