# Experiment framework: текущая реализация

## Назначение

Первый реализованный вертикальный срез находится в `experiments/`. Он безопасно
подготавливает RAGTruth-данные, создаёт no-gold input, запускает детекторы через один
runner и записывает проверяемый prediction archive. Он сознательно не выполняет
скачивание датасета, real-model inference, DataSphere Job или gateway call сам по себе.

Это соответствует текущему режиму работы: код можно проверить локально на fixtures/mock,
а live run будет выполняться позднее в DataSphere с закреплёнными secrets и revisions.

## Что реализовано

### Controlled shared-KGGen track, reusable graph caches and three-way typing scaffold

`controlled_shared_kggen_response_v1` remains an explicit additional track. Its
scientific comparison shares the answer graph between HalluGraph and GraphEval; the
current runner additionally seals the complete `(G_context, G_query, G_answer)` bundle
before either detector so that all KGGen provenance is fixed. GraphEval consumes its
common answer graph and raw context, whereas HalluGraph consumes all three graphs. The
existing `kggen_untyped_adaptation` construction is unchanged.

`controlled_shared_all_graphs_three_way_stub_v1` adds the planned three-way
comparison without altering KGGen: the framework materializes one immutable
`(G_context, G_query, G_answer)` bundle before all detector calls, then records its
identity on GraphEval, untyped HalluGraph and typed-HalluGraph rows. The typed variant
currently uses an all-`unknown`, score-preserving placeholder. It defines and archives
the input/output contract, but contains no model, prompt or policy for type induction.
See `docs/dynamic-typing-experiment-infrastructure.md`.

`experiments/shared_graphs.py` validates historical `hallu-kg-cache-v2` sources,
detects corrupt/conflicting entries and supports cache-only/read-through policies.
`python -m experiments.cli cache inspect` performs a read-only structural and optional
no-gold coverage audit. Details: `docs/shared-kggen-controlled-track.md` and
`docs/graph-cache-reuse.md`.

`examples/mock_shared_kggen_one_instance.py` provides a response-ID two-pass probe. Its
first offline pass materializes common graphs and its second proves `cache_only` replay
from the same configurable `cache_root`; both use FakeKGGen/FakeNLI and seal archives.

The former DataSphere shared-KGGen mock job has been replaced by the separately named
`shared-kggen-one-instance-controlled-live` path. It uses real controlled KGGen/HHEM,
requires the Project secret at runtime, and has dependency-injected offline tests only;
no new live Job has been submitted.

### 1. Общий framework-facing контракт

`experiments/contracts.py` использует уже проверенные `DetectionInput` и
`DetectionResult` из `graph_eval.types`, добавляя:

- `DetectorProtocol` для generic runner;
- allowlist metadata;
- рекурсивный запрет gold/label/span/quality/due-to-null/implicit-true полей;
- единый serializable prediction record.

Семантика детекторов не менялась. Полная миграция dataclasses в независимый
`detector_contracts/` остаётся следующим refactor-этапом: текущая обёртка уже изолирует
framework callers от того, где контракт физически объявлен.

### 2. RAGTruth local data path

`experiments/datasets/ragtruth.py` умеет работать с уже локально доступными
`source_info.jsonl` и `response.jsonl`:

- проводит audit IDs, offsets, labels, splits и распределений;
- создаёт checksum/versioned data manifest;
- строит source-level deterministic sample manifest без чтения gold для selection;
- materializes `instances.no_gold.jsonl` и отдельные `gold/response_gold.jsonl`,
  `gold/gold_spans.jsonl`;
- сохраняет raw hashes, context/query/response hashes и RAGTruth metadata;
- использует существующий task-native context builder для QA/Summary/Data2txt.

`data fetch` реализован, но не вызывался: он принимает только exact 40-character Git commit
SHA, отвергает `main`, скачивает два опубликованных JSONL во временные файлы и затем запускает
audit/checksum. Текущий агент не запускал эту команду и не загружал данные. Локальный data path
по-прежнему обеспечивается аргументами `--source-info` и `--responses`.

### 3. Immutable prediction archive

`experiments/artifacts.py` создаёт `runs/<run_id>/` со следующими рабочими частями:

```text
run_manifest.json
schema_versions.json
instances.no_gold.jsonl
stages/stage_calls.jsonl
predictions/raw_predictions.jsonl
predictions/paired_predictions.jsonl
shared_graphs/graph_index.jsonl
shared_graphs/bundles.jsonl
typing/type_registries.jsonl
typing/type_annotation_bundles.jsonl
prediction_seal.json
payloads/sha256/
reports/
```

JSON/JSONL writes атомарны через temporary file + `os.replace`; seal записывает checksums
prediction/stage файлов. Archive validator проверяет их при повторном чтении.

### 4. Paired runner

`experiments/runner.py`:

- читает только no-gold JSONL и повторно применяет leakage guard;
- подаёт один и тот же `DetectionInput` всем выбранным detectors;
- изолирует per-item exception как `failed`, не превращая его в score;
- пишет stage calls, raw predictions и paired table;
- поддерживает безопасный resume по `(variant, response_id)`;
- разрешает seal только при полном покрытии ожидаемых IDs.

### 5. Offline mock demonstration

`examples/mock_experiment_demo.py` — отдельный воспроизводимый пример без ключей,
датасета или сети. Он использует два явно маркированных mock detectors, создаёт archive
и печатает Unicode-таблицу рисков. Запуск:

```bash
python examples/mock_experiment_demo.py --output-root examples/mock_output
```

Или:

```bash
python -m experiments.cli demo --output-root examples/mock_output
```

Выводы mock-демо не являются метриками HalluGraph или GraphEval; он проверяет только
проводку framework и наглядный формат артефактов.

### 6. Post-seal evaluation baseline

`experiments/evaluation.py` уже реализует безопасный минимальный evaluation path:

- присоединение response-level gold только после `prediction_seal.json`;
- complete-case coverage, AUROC, AUPRC, precision/recall/F1, specificity и balanced accuracy;
- явный threshold comparator и confusion matrix;
- отдельные `evaluation/predictions_with_gold.jsonl` и `evaluation/metrics.jsonl`.

Порог передаётся явно в CLI только для локального plumbing. До live test он должен быть
заменён ссылкой на train-only frozen threshold manifest.

## Доступные безопасные CLI-команды

Все эти команды работают только с уже доступными локальными файлами либо mock data:

```bash
python -m experiments.cli data audit \
  --source-info <local/source_info.jsonl> \
  --responses <local/response.jsonl> \
  --revision <pinned-sha> --output <data_manifest.json>

python -m experiments.cli sample create \
  --source-info <local/source_info.jsonl> \
  --responses <local/response.jsonl> \
  --data-manifest <data_manifest.json> --split train --seed 42 \
  --n-sources 20 --output <sample_manifest.json>

python -m experiments.cli sample materialize \
  --source-info <local/source_info.jsonl> \
  --responses <local/response.jsonl> \
  --data-manifest <data_manifest.json> \
  --sample-manifest <sample_manifest.json> --output-dir <subset-dir>

python -m experiments.cli archive validate --runs-root <runs-root> --run-id <run-id>
```

## Реальные detector factories

`experiments/detectors.py` содержит:

- `build_grapheval_fake()` — реальный GraphEval facade с FakeExtractor/FakeNLI;
- `build_hallugraph_fake()` — реальный HalluGraph adapter с FakeKGGen/DictEmbedder;
- `build_real_detectors(...)` — только конструктор для будущего явного live-конфига.

Ни одна CLI-команда первой версии не вызывает `build_real_detectors`; это deliberate safety
barrier до согласованного DataSphere preflight.

## Что ещё не реализовано

Следующие пункты остаются обязательными до scientific live run и описаны подробно в
`docs/experiment-framework-implementation-spec.md`:

1. pinned HTTP fetch/import lifecycle и full raw-store layout;
2. `ArtifactSink` внутри GraphEval/HalluGraph с raw graphs, claims, candidate lattice и
   per-NLI trace;
3. train-only grouped tuning, threshold manifests и policy replay;
4. gold join, metrics, source-cluster bootstrap, calibration, localization и reports;
5. blind audit workflow, cost/stability analysis, router/hybrid tables;
6. модельный агент динамической типизации, его промпт, версия политики, богатые
   node IDs и собственно изменение метрики HalluGraph; контракт и score-preserving
   трёхсторонняя заглушка уже реализованы;
7. DataSphere execution backend, gateway manifest gate и cache-only live replay;
8. Parquet/zstd large-artifact compaction and public redacted export.

## Проверки

Offline tests расположены в `tests/experiments/` и не требуют RAGTruth, OpenAI, gateway,
torch или secrets. Они проверяют gold isolation, deterministic sampling, archive sealing,
checksums и resume.

Отдельный mock-тест `tests/experiments/test_three_way_dynamic_typing.py` проверяет
общий набор из трёх KGGen-графов, матрицу из трёх вариантов, архивные строки
типизации и cache-only replay с нулём вызовов KGGen.

В этой рабочей среде pytest может обнаружить внешний пользовательский `.git` выше корня
репозитория. Поэтому framework tests запускаются с локальным config:

```bash
python -m pytest -c pytest.framework.ini -q
```

Перед первым live run нужно пройти все gates раздела 29.7 основного ТЗ. До этого
разрешены только unit tests и mock/fake checks.
