# Interactive run viewer

`scripts/render_run_viewer.py` converts one local, no-gold run into a single
browser-openable `typing-run-viewer.html`. It is fully self-contained: it does not start a
server, fetch a resource, call an LLM or contact HHEM.

The viewer is designed for inspection rather than aggregate evaluation:

- switches between context, query and answer;
- highlights extracted entity mentions and lets the reader select a sentence;
- follows an evidence span into the source text;
- shows source and answer assignments alongside the frozen type tree;
- exposes definition, parent type, evidence level and NLI rationale without crowding the
  initial screen.

Every new `run-fixture` run now writes `input_snapshot.json` beside the existing source
registry and answer annotations. It contains only the source/answer inputs used by this
no-gold command, never a gold label or secret. Render a new run with:

```powershell
$env:PYTHONPATH = "src"
python scripts\render_run_viewer.py --run runs\live-test
```

For a run created before input snapshots existed, supply the same no-gold fixture that
created it:

```powershell
python scripts\render_run_viewer.py `
  --run runs\live-test `
  --fixture examples\dynamic_typing_20.no_gold.jsonl
```

The command writes `runs\live-test\<case-id>\typing-run-viewer.html`. Open that file in a
browser. With several successful cases, also give `--case <case-id>`.
