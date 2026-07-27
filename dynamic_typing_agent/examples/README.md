# Example corpus

`dynamic_typing_20.no_gold.jsonl` is the only file permitted as agent input. It contains
raw context/query/response text and small immutable graph fixtures.

`dynamic_typing_20.expectations.jsonl` is a human-review oracle for later tests and
qualitative inspection. Production/local agent loaders must never open it. Case IDs link
the files only after an agent run has been sealed.

The expectations describe semantic targets rather than exact generated labels. Dynamic
type labels may vary while hierarchy, roles, abstention and NLI behavior remain correct.

