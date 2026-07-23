# Interactive run viewer

The viewer is a local, dependency-free inspection application generated from sealed
no-gold run artifacts. It does not call an LLM, HHEM, KGGen or a web service. All pages
work directly through `file://`.

## Layout

`<run>/viewer/index.html` is the run dashboard. It shows every case, status, input mode,
graph size, number of types and NLI calls. Search and filters work locally. A case link
opens `<run>/viewer/cases/<case-id>/index.html`.

Each case page contains:

- tabs for context, query and answer when an answer exists;
- sentence-level source text with entity mentions colored by assigned type;
- a real canvas knowledge graph with labeled directed edges, drag, pan and zoom;
- the same type color on graph nodes, text mentions, assignments and hierarchy entries;
- a searchable hierarchical type dictionary and a role-specific entity list;
- entity details with reasons, evidence spans, types and related NLI results;
- a chronological source/answer event log;
- human-readable event descriptions such as “the model proposed a new type”, “the type
  was assigned”, “the hierarchy change was rejected” or “the types were merged”;
- explicit “model answered/proposed” and “NLI decision” sections;
- expandable raw stage input, output and complete event JSON for deep debugging.

Selecting an entity in text, the canvas graph or the entity list updates every related
surface. Selecting a type highlights all matching graph nodes and assignments. Selecting
an evidence span moves to its source role.

## Generate or rebuild

The preferred `hallugraph-type-agent test` command generates the viewer automatically.
To rebuild a viewer from an existing compatible run:

```powershell
$env:PYTHONPATH = "src"
python scripts\render_run_viewer.py --run runs\my-run
```

The renderer accepts both the canonical `run_manifest.json` and earlier list-shaped
`summary.json` files, provided each successful case contains `input_snapshot.json`.

## Offline asset contract

Generated pages use only files under the run's `viewer/` directory:

```text
viewer/
  index.html
  run-data.js
  assets/
    styles.css
    dashboard.js
    case.js
  cases/
    <case-id>/
      index.html
      data.js
```

There are no remote fonts, libraries, `fetch` calls or dynamic file reads. JSON is emitted
as JavaScript data with `<` and Unicode line separators escaped, so source content cannot
close or inject a script element.
