# Техническое задание: экспериментальный фреймворк GraphEval × HalluGraph на RAGTruth

## 0. Статус документа и назначение

Этот документ является техническим заданием для агента, который будет реализовывать
экспериментальный фреймворк в репозитории `hallu_smiles`.

Документ подготовлен в ветке `codex/experiment-framework-spec`, созданной от `sasha`
на коммите `2c1d3c5`. На исходной ветке уже находятся:

- HalluGraph-KGGen в `src/` и `run.py`;
- GraphEval как самостоятельный пакет в `graph_eval/`;
- адаптер HalluGraph к общему контракту в `detector_adapters/`;
- offline fake-backends и тесты обоих методов;
- инфраструктура Cloud Run gateway и заготовки DataSphere.

Следующий агент должен реализовать новый слой `experiments/`, не заменяя и не
дублируя научную логику существующих детекторов.

Нормативные источники требований:

1. `COMPARISON_PROTOCOL.md` — научные вопросы, треки, метрики и границы выводов.
2. `RAGTRUTH_ONE_INSTANCE_PROTOCOL.md` — полный путь одного объекта.
3. `EXPERIMENT_ARTIFACTS_AND_LOGGING_SPEC.md` — максимальный контракт артефактов.
4. `docs/graph_eval_integration.md` — фактический интерфейс реализованных методов.
5. `docs/vertex-datasphere-team-runbook.md` и
   `docs/cloud-run-datasphere-agent-tutorial.md` — будущий облачный запуск.

Если требования расходятся, максимальный logging contract определяется
`EXPERIMENT_ARTIFACTS_AND_LOGGING_SPEC.md`, а primary endpoints и ограничения
научного вывода — `COMPARISON_PROTOCOL.md`.

## 1. Цель и критерий успеха

Нужно построить воспроизводимый фреймворк, который:

1. получает точную закреплённую версию RAGTruth;
2. создаёт детерминированные полные или выборочные manifests;
3. физически отделяет detector input от gold-разметки;
4. запускает HalluGraph и GraphEval через одинаковый интерфейс на одних объектах;
5. сохраняет все промежуточные данные, необходимые для причинного анализа различий;
6. позволяет пересчитывать thresholds, policies, метрики, срезы и часть абляций без
   повторных LLM/NLI-вызовов;
7. поддерживает локальный offline smoke, локальный live run и будущий запуск в
   DataSphere по одному и тому же научному пути;
8. выдаёт проверяемый архив, из которого восстанавливается путь
   `raw input → extraction → verification → score → decision → metric`;
9. не допускает использования test или gold при inference/tuning;
10. масштабируется на новые детекторы, экспериментальные треки и hybrid/router-анализ.

Эксперимент считается завершённым не тогда, когда построена одна таблица метрик, а
тогда, когда архив прошёл schema, lineage, checksum и leakage validation.

## 2. Научные ограничения, которые код обязан обеспечивать

### 2.1. Формулировка задачи

Проверяется closed-context groundness: поддержано ли утверждение ответа только данным
RAG-контекстом. Внешние знания, внешние документы и внешние KG запрещены.

### 2.2. Группировка

`source_id` — неделимая статистическая группа:

- ответы одного `source_id` нельзя разводить между development folds;
- sampling по умолчанию выполняется на уровне `source_id`;
- bootstrap выполняется по `source_id`, а не по отдельным responses;
- paired analysis использует только совпадающие `response_id` одного manifest.

### 2.3. Test discipline

- Публичный RAGTruth `train` используется для разработки и group-CV.
- Публичный RAGTruth `test` используется только после freeze.
- Подвыборка test должна быть зафиксирована до просмотра predictions.
- Никакие модели, prompts, thresholds, alpha, tau, policies или primary slices не
  выбираются по test.
- Любой post-hoc срез маркируется `exploratory=true`.

### 2.4. Направление score

Для каждого метода framework хранит `raw_score`, где большее значение означает более
вероятную галлюцинацию. HalluGraph дополнительно сохраняет fidelity `CFI`, где направление
обратное. Нельзя заменять один score другим или терять исходные компоненты.

### 2.5. Неуспех не является предсказанием

`empty_graph`, parse failure, NLI failure, truncation, cache-only miss и transport failure
не превращаются молча в factual/hallucinated. Это отдельные состояния с заранее
замороженной policy и обязательным sensitivity analysis.

## 3. Обязательные экспериментальные треки и честное именование

Framework должен понимать следующие `comparison_track`:

| Track | Смысл | Статус при старте реализации |
|---|---|---|
| `kggen_untyped_adaptation` | Текущий HalluGraph-KGGen против текущего GraphEval | Первый исполнимый primary track |
| `faithful_replication` | Авторские extractor/verifier каждого метода | Нельзя заявлять готовым без spaCy+SLM HalluGraph и parity evidence |
| `controlled_shared_answer_graph` | Один answer graph, разные verifier-механизмы | Реализован как `controlled_shared_kggen_response_v1`; см. `shared-kggen-controlled-track.md` |
| `controlled_shared_all_graphs` | Общие answer/context/query extraction artifacts | Отдельный controlled этап |
| `typed_graph_ablation` | B0–B4 абляции типов | Расширение после baseline |
| `selective_nli_hybrid` | Structural method + selective NLI | Не primary для первого сравнения |
| `exploratory` | Явно исследовательский запуск | Никогда не смешивать с confirmatory |

Preflight должен отвергать `faithful_replication`, если конфигурация фактически использует
`kggen_untyped`. Нельзя получать научно неверную маркировку простым изменением строки в YAML.

## 4. Архитектурные решения

### ADR-1. Не копировать реализации методов

HalluGraph остаётся в `src/`/`run.py`, GraphEval — в `graph_eval/`. Новый framework
импортирует их через adapters. Копии алгоритмов внутри `experiments/` запрещены.

### ADR-2. Нейтральный zero-dependency контракт

Текущий контракт находится в `graph_eval.types`, из-за чего HalluGraph adapter формально
зависит от GraphEval. Нужно создать нейтральный top-level пакет `detector_contracts/`:

```text
detector_contracts/
  __init__.py
  types.py
  protocol.py
  statuses.py
```

В него переносятся без изменения семантики:

- `DetectionInput`;
- `DetectionResult`;
- общие статусы;
- JSON-compatible type aliases;
- runtime-checkable `DetectorProtocol`.

`graph_eval.types` должен re-export старых имён для обратной совместимости. Тесты
GraphEval и HalluGraph parity обязаны продолжить проходить.

### ADR-3. Оркестратор не знает внутренностей метода

Оркестратор работает только с:

```python
class DetectorProtocol(Protocol):
    method_name: str
    variant_name: str
    def predict(self, item: DetectionInput) -> DetectionResult: ...
```

Построение детекторов выполняет registry/factory. Method-specific конфигурация и
artifact exporters находятся в adapters, а не в generic runner.

### ADR-4. Prediction и evaluation — разные trust zones

Prediction-процесс читает только `instances.no_gold.jsonl`. Evaluation-процесс отдельно
читает sealed predictions и `gold/`. Gold не просто удаляется из dataclass — он находится
в другом физическом файле и присоединяется после создания prediction seal.

### ADR-5. Артефакты — часть результата, а не debug logging

Каждый detector adapter получает `ArtifactSink`/observer и публикует versioned records.
Если обязательный artifact невозможно записать, run становится invalid/failed; нельзя
оставить score, путь к которому нельзя восстановить.

### ADR-6. Один scientific path для local и DataSphere

Execution backend может различаться, но dataset adapter, manifests, detector configs,
runner, schemas, evaluation и reporting одинаковы. Отдельный «облачный алгоритм» запрещён.

### ADR-7. Raw inference отделён от дешёвого replay

Дорогие стадии создают immutable extraction/NLI artifacts. Thresholding, aggregation,
policy replay, slices, bootstrap и reports читают сохранённые данные и не вызывают модели.

## 5. Предлагаемая структура репозитория

```text
hallu_smiles/
  detector_contracts/                 # нейтральный общий интерфейс
  detector_adapters/
    hallugraph_adapter.py
    grapheval_adapter.py               # явная симметричная обёртка/factory
    registry.py
    instrumentation/
  experiments/
    __init__.py
    cli.py
    config/
      models.py
      load.py
      validate.py
      freeze.py
    datasets/ragtruth/
      download.py
      raw_store.py
      schema.py
      audit.py
      canonicalize.py
      serialize.py
      sampling.py
      materialize.py
      gold.py
    manifests/
      ids.py
      run.py
      sample.py
      threshold.py
    orchestration/
      planner.py
      runner.py
      state.py
      resume.py
      execution_backend.py
      preflight.py
    artifacts/
      sink.py
      schemas.py
      jsonl.py
      parquet.py
      payload_store.py
      lineage.py
      checksums.py
      validate.py
      redaction.py
    tuning/
      grouped_cv.py
      thresholds.py
      hallugraph.py
      calibration.py
    evaluation/
      join_gold.py
      metrics.py
      bootstrap.py
      paired.py
      slices.py
      localization.py
      failures.py
      stability.py
      cost.py
      policy_replay.py
      oracle.py
    audit/
      sampling.py
      export.py
      import_annotations.py
      adjudication.py
    reporting/
      tables.py
      plots.py
      report.py
    schemas/                           # versioned JSON Schemas
    testing/                           # fake builders/fixtures
  configs/experiments/
    offline-smoke.yaml
    kggen-vs-grapheval-dev.yaml
    kggen-vs-grapheval-test.yaml
    cache-only-replay.yaml
  scripts/
    run_experiment.sh
    validate_experiment_archive.py
  datasphere/jobs/
    experiment-cpu.template.yaml       # добавить на облачном этапе
  docs/
    experiment-framework-implementation-spec.md
  requirements.experiments.txt
  tests/experiments/
```

Не следует преждевременно перепаковывать весь существующий `src/`: adapters должны
изолировать его историческую структуру.

## 6. Общий контракт

### 6.1. `DetectionInput`

Разрешены только:

```text
response_id
source_id
context
query
response
metadata_allowlisted
```

Metadata allowlist первой версии:

```text
task
source_dataset
generator_model
generator_temperature
context_document_ids
context_document_order
dataset_record_id
```

Запрещены рекурсивно и по шаблону:

```text
label, labels, gold, span, label_type, due_to_null, implicit_true,
quality, meta, target, y, response_label
```

Перед каждым вызовом detector выполняется runtime leakage check. Неизвестное metadata
поле отклоняется, а не молча пропускается.

### 6.2. `DetectionResult`

Общий обязательный минимум:

```text
response_id, source_id
method, variant
raw_score
score_name, score_semantics, score_direction, formula_version
components
flagged_unit_ids
status
failure
usage
artifact_refs
```

Инварианты:

- `status != ok` означает `raw_score = null`;
- `failed` требует структурированную причину;
- score находится в `[0,1]`, если метод декларирует unit interval;
- IDs результата совпадают с input;
- все `artifact_refs` разрешаются внутри run archive;
- результат не содержит gold-полей до evaluation.

### 6.3. Стабильные IDs

Не использовать текст как внешний ключ. IDs строить детерминированно из namespace,
parent IDs, ordinal и canonical hash. Random UUID допустим для `run_id`, но IDs внутри
run должны воспроизводиться при cache-only replay.

Минимальные namespaces:

```text
dataset_record_id, method_run_id, stage_call_id, graph_id, node_id, edge_id,
claim_id, candidate_id, alignment_id, nli_call_id, prediction_id, audit_id
```

## 7. RAGTruth data subsystem

### 7.1. Канонический источник

Использовать официальный репозиторий:

```text
https://github.com/ParticleMedia/RAGTruth
```

Скачивать полные `dataset/source_info.jsonl` и `dataset/response.jsonl`, поскольку общий
размер мал. URL обязательно формировать с exact Git commit SHA, а не с `main`.

### 7.2. Команды получения данных

```bash
python -m experiments.cli data fetch \
  --revision <40-char-commit-sha> \
  --data-root data/ragtruth

python -m experiments.cli data import \
  --source-info /path/source_info.jsonl \
  --responses /path/response.jsonl \
  --declared-revision <sha-or-internal-id>

python -m experiments.cli data audit \
  --dataset data/ragtruth/raw/<revision>
```

`fetch` обязан:

1. скачать во временные уникальные файлы;
2. проверить HTTP status и ненулевой размер;
3. вычислить SHA-256 и число строк;
4. провести минимальный parse/schema audit;
5. атомарно переместить файлы;
6. записать `data_manifest.json` и `checksums.sha256`;
7. при повторе принимать существующий dataset только после checksum validation;
8. не исправлять исходные строки и newline convention.

### 7.3. Layout данных

```text
data/ragtruth/
  raw/<revision>/
    source_info.jsonl
    response.jsonl
    data_manifest.json
    checksums.sha256
  derived/<dataset_hash>/<ragtruth_adapter_version>/
    source_index.parquet
    response_index.parquet
    schema_audit.json
    canonical_records/
  subsets/<sample_manifest_hash>/
    instances.no_gold.jsonl
    gold/response_gold.jsonl
    gold/gold_spans.jsonl
    sample_manifest.json
    materialization_manifest.json
    checksums.sha256
```

Raw data, derived indices и subsets не коммитить.

### 7.4. Raw, canonical, derived

Для каждой JSONL-строки сохранить:

- exact raw bytes/string и line number;
- raw record SHA-256;
- parsed object;
- canonical JSON с фиксированной UTF-8/sort/separator policy;
- derived поля отдельно.

Canonicalization никогда не изменяет raw. Hash cache input строится из явно
версионированного canonical representation.

### 7.5. Data audit

Audit обязан проверить и сохранить всё из раздела 4 artifact spec, включая:

- уникальность source/response IDs;
- orphan responses;
- source IDs, пересекающие split;
- число responses на source;
- известные task/model/quality/label types;
- корректность gold offsets и совпадение `response[start:end] == text`;
- overlapping spans;
- encoding/newline convention;
- распределения по task, source dataset, split, model, temperature, quality;
- `due_to_null` и `implicit_true`;
- точные bytes/line counts/checksums.

Critical audit errors блокируют materialization; предупреждения сохраняются и требуют
явного `--allow-warning CODE`.

### 7.6. Построение context/query

- QA: `query = question`, `context = passages`, порядок и raw separators сохраняются.
- Summary: `context = article`, `query = null`.
- Data2txt: исходный JSON сохраняется, а детерминированный serializer создаёт derived
  context и обратимый `serializer_trace` от каждого предложения к JSON path.

Нужно различать `null`, `false`, `true`, строку `"no"`, missing key и empty string.
Текущий serializer в `src.data` можно переиспользовать как baseline, но расширить trace
без изменения его текстового output; parity проверить golden tests.

QA passages должны получить стабильные `document_id`, `rank` и offsets. Разбиение должно
быть обратимым и не менять итоговый `context_raw`.

## 8. Sampling и manifests

### 8.1. Принцип

Dataset скачивается полностью, но inference выполняется по immutable sample manifest.
Scientific run не принимает произвольный `--limit N`, потому что первые N строк дают
order-biased выборку. `--limit` допустим только при `run_purpose=smoke`.

### 8.2. Поддерживаемые sampling modes

```text
all                         # весь указанный split
source_random               # случайная выборка source_id
source_stratified           # по неголдовым strata
explicit_source_ids
explicit_response_ids       # только smoke/debug, с явной маркировкой
manifest_replay
audit_stratified            # создаётся в evaluation trust zone
```

Параметры:

```text
split, task, source_dataset, generator_model, generator_temperature,
quality_filter, n_sources, fraction, seed, include_all_responses_per_source,
one_response_per_source, response_selection_policy
```

Для primary development/test sample default:

- sampling unit — `source_id`;
- stratification — task/source dataset/generator model, без gold;
- после выбора source включаются все его responses данного split;
- manifest сортируется детерминированно;
- сохраняется selection probability.

`one_response_per_source` разрешить для дешёвых pilots, но такой run маркируется как pilot,
а его метрики нельзя выдавать за оценку полной response population.

### 8.3. Sample manifest

Обязательные поля:

```text
sample_manifest_version
sample_manifest_id/hash
dataset_revision и raw checksums
created_at_utc
purpose
selection_stage
sampling_unit
algorithm/version
seed
filters
stratification
include_all_responses_per_source
selected_source_ids
selected_response_ids
selection_order
selection_probability по stratum/record
counts by split/task/source/model
gold_used_for_selection
prediction_used_for_selection
```

Для test primary должно быть:

```text
gold_used_for_selection = false
prediction_used_for_selection = false
```

### 8.4. Предопределённые purposes

```text
unit_fixture       # синтетика, без сети
smoke              # 2–5 real responses
probe              # несколько source из всех tasks
development        # train subset
threshold_tuning   # train group-CV
stability          # заранее выбранный train/audit subset
manual_audit       # blind audit sample
final_test_sample  # frozen test subset
full_test          # поддерживается, но не обязателен проекту
```

### 8.5. Materialization и gold isolation

Из одного sample manifest создать два физически независимых набора:

```text
instances.no_gold.jsonl
gold/response_gold.jsonl
gold/gold_spans.jsonl
```

`instances.no_gold.jsonl` содержит raw inputs, hashes, provenance и неголдовые metadata.
Gold files не должны находиться в search path prediction workers. Prediction CLI не
принимает путь к gold даже опционально.

## 9. Конфигурация эксперимента

### 9.1. Один resolved config

Входной YAML проходит strict validation, default resolution и freeze в
`resolved_config.yaml/json`. Неизвестные поля — ошибка. CLI overrides фиксируются отдельно.

Основные секции:

```yaml
experiment:
dataset:
sample:
tracks:
methods:
  hallugraph:
  grapheval:
tuning:
thresholds:
failure_policy:
empty_graph_policy:
artifacts:
evaluation:
bootstrap:
stability:
reporting:
runtime:
cache:
security:
```

### 9.2. Freeze manifest

Перед test команда `freeze` создаёт immutable manifest со всем списком раздела 26 artifact
spec. Test runner принимает только freeze manifest hash и отказывается работать с dirty,
неполной или изменённой конфигурацией без `exploratory=true`.

### 9.3. Track capability validation

Каждый adapter публикует capabilities:

```text
extractor_family
answer_graph_shareable
context_graph_available
entity_types_available
relation_types_available
candidate_lattice_available
raw_logits_available
offset_provenance_level
supports_cache_only
```

Preflight сопоставляет capabilities требованиям track и не допускает ложный режим.

## 10. Run identity и state machine

### 10.1. Идентификаторы

```text
experiment_id     # общий план сравнения
run_id            # конкретное исполнение manifest/config
method_run_id     # один method+variant внутри run
parent_run_id     # replay/resume/stability relation
```

Рекомендуемый `run_id`: UTC timestamp + короткий hash resolved config/sample/code, без
секретов и случайной неоднозначности.

### 10.2. Состояния

```text
created
preflight_passed
running_predictions
predictions_complete
predictions_sealed
gold_joined
analysis_complete
archive_validated
completed
failed
cancelled
```

Transitions атомарно записываются в `run_manifest`. `gold_join` запрещён до
`predictions_sealed`. После seal raw predictions не изменяются.

### 10.3. Prediction seal

Seal содержит:

- hash каждого prediction/stage/artifact файла;
- expected и observed response IDs;
- counts по status;
- code/config/sample hashes;
- timestamp завершения prediction;
- `gold_access_state=hidden`.

## 11. Варианты запуска

### 11.1. По вычислительному режиму

```text
offline_fake       # deterministic fake methods, CI и разработка
local_live         # реальные backends локально
datasphere_live    # тот же runner внутри Job
cache_only_replay  # 0 model/network calls
artifact_replay    # thresholds/analysis только из archive
```

### 11.2. По охвату методов

```text
single_method
paired_both
one_method_then_resume_other
variant_matrix
```

Primary comparison должен быть `paired_both` либо два method runs с одним и тем же
sample manifest hash. Pairing validator проверяет точное совпадение ID.

### 11.3. По стадиям

```text
prepare-data
preflight
extract
verify/score
predict
seal
tune
threshold
join-gold
evaluate
analyze
report
validate-archive
```

Нужно уметь запускать весь DAG и отдельный downstream stage. Downstream stage всегда
проверяет hashes upstream artifacts.

### 11.4. Cold/warm профили

```text
fully_cold
cold_context_graph
cached_context_graph
fully_warm
cache_only_replay
```

Очистка cache не должна быть частью обычной команды run. Cold run использует новый
namespace/directory, чтобы не удалять пользовательские данные.

### 11.5. Resume

Resume выполняется по составному ключу `(method_run_id, response_id)`, а не только
`response_id`. Перед пропуском строки проверяются input/config/code hashes и целостность
всех обязательных artifact refs. Повреждённая/частичная запись не считается completed.

## 12. Оркестрация

### 12.1. DAG

```text
fetch/import → data audit → sample manifest → materialize no-gold/gold
             → resolve/freeze config → preflight
             → HalluGraph method run ┐
             → GraphEval method run  ├→ prediction seal
                                     ┘
             → train-only tune/freeze thresholds
             → gold join → metrics/slices/paired/localization
             → disagreements/audit/cost/stability
             → report → archive validation
```

### 12.2. Порядок и concurrency

- Input order фиксируется manifest.
- Worker scheduling может различаться, output compaction сортирует по stable keys.
- Один worker не пишет напрямую в общий JSONL без single-writer queue/partitioning.
- Parallel method runs не должны перегружать общий gateway; concurrency задаётся отдельно
  по component/provider.
- Batching NLI не должно менять порядок IDs или scores.
- Concurrency, batch size и request order сохраняются в manifest.

### 12.3. Fail-fast и continue

Конфигурационные ошибки, leakage, schema mismatch, 4xx auth/config, несовпадение dataset
hash и неразрешимые artifact refs — run-level fail-fast.

Per-instance extractor/NLI failure сохраняется как отдельный result и по заранее выбранной
policy позволяет продолжить другие instances. Доля failures выше установленного preflight
лимита делает run invalid даже при наличии части predictions.

## 13. Интеграция GraphEval

Adapter обязан сохранить существующую научную формулу:

```text
response-only triples
premise = context
hypothesis = verbalized triple
p_unsupported = 1 - p_consistent
H = max(p_unsupported)
paper decision: H > 0.5
```

Дополнительно instrument/export:

- exact answer input hash и доказательство, что extractor не видел context/query/gold;
- prompt/schema/model/gateway fingerprints;
- raw extractor output и parse/repair trace;
- все valid/invalid/duplicate triples;
- answer offsets и posthoc alignment status;
- premise/hypothesis hashes, tokens, truncation;
- HHEM revision, label order, raw logits/probabilities, если backend их предоставляет;
- per-claim scores/decision;
- response aggregation trace;
- NLI/extraction cache hits и calls.

Если текущий HHEM adapter возвращает только `p_consistent`, расширение должно быть
обратно совместимым: score не меняется, дополнительные raw fields добавляются через
trace object/observer.

## 14. Интеграция HalluGraph

Adapter обязан продолжать вызывать те же `run.build_refgraph` и
`src.metrics.score_response`; parity с `run.build_rows` — блокирующий тест.

Сохранять:

- raw `G_C`, `G_Q`, `G_A` до упрощения;
- graph nodes/edges/clusters и provenance;
- normalized graph;
- все entity candidates, accepted и top-k rejected;
- все edge candidates, directions и rejection reasons;
- selected matches и runner-up margins;
- EG/RP numerator/denominator и exclusions;
- `RP_defined`, edge-aware policy;
- alpha, tau_e, tau_r, CFI, H и decision trace;
- strict/support/support-critical components только для реально запущенного mode;
- calls/cache/latency по стадиям.

Текущие `RefGraph.match_entity/align_relation` не сохраняют весь candidate lattice. Их
следует расширять optional trace callback/return detail так, чтобы default path и численные
результаты не изменились. Нельзя заново реализовать matching в framework.

## 15. Controlled mechanism tracks

### 15.1. Общий graph IR

Нужен нейтральный immutable representation:

```text
GraphArtifact
NodeArtifact
EdgeArtifact
ClusterArtifact
ProvenanceSpan
ExtractionTrace
```

IR хранит raw и normalized значения, но не навязывает типы. Метод может читать один и тот
же answer graph artifact без нового extraction.

### 15.2. Shared answer graph

В `controlled_shared_answer_graph`:

- answer extraction выполняется один раз;
- GraphEval получает triples из IR;
- HalluGraph получает тот же набор nodes/edges;
- различаются verification/aggregation paths;
- artifact hash общего графа одинаков у обоих method runs.

### 15.3. Shared all graphs

В `controlled_shared_all_graphs` общий extractor/config применяется к answer/context/query.
GraphEval по-прежнему не должен передавать context в answer extractor; context используется
только NLI verifier. Track проверяет механизм, а не faithful package performance.

## 16. Artifact subsystem

### 16.1. Run archive

Реализовать структуру из `EXPERIMENT_ARTIFACTS_AND_LOGGING_SPEC.md` без потери полей:

```text
runs/<run_id>/
  run_manifest.json
  schema_versions.json
  environment.json
  resolved_config.yaml
  data_manifest.json
  sample_manifest.json
  instances.jsonl
  prediction_seal.json
  gold/
  stages/
  graphs/
  grapheval/
  hallugraph/
  predictions/
  evaluation/
  router/
  audit/
  payloads/sha256/
  reports/
  checksums.sha256
```

### 16.2. Форматы

- JSON/JSONL — manifests, compact records, human inspection.
- Parquet — bootstrap replicates, large candidates/features, analysis tables.
- `.json.zst` — content-addressed heavy payloads.
- Markdown/PNG/SVG/CSV — reports; source metric table hashes обязательны.

JSONL first implementation допустим, но schema и IDs должны позволять без потерь
компактировать его в Parquet.

### 16.3. `ArtifactSink`

Минимальный API:

```python
emit(record_type, payload, schema_version, parent_refs)
put_payload(payload_or_bytes, media_type, retention_class) -> ArtifactRef
flush_instance(response_id)
checkpoint()
finalize()
```

Sink добавляет run/method IDs, validates schema, redacts secrets, пишет atomically и
возвращает content-addressed refs. Detector не должен сам придумывать пути run archive.

### 16.4. Atomicity

- Single JSON пишет `.<name>.<uuid>.tmp`, flush/fsync по возможности, затем `os.replace`.
- Content store сначала проверяет hash existing payload.
- Concurrent writers используют уникальные temp files и single-writer index.
- JSONL checkpoint содержит last complete offset; оборванная строка не считается valid.
- Finalization сортирует/валидирует records и только затем пишет seal.

### 16.5. Schema evolution

Каждый record имеет:

```text
record_type
schema_version
producer_component
producer_version
run_id
method_run_id
gold_access_state
```

Schemas versioned; breaking change повышает major. Нужны migrations только для реально
поддерживаемых старых archives; неизвестная major version — явная ошибка.

### 16.6. Payload retention

Во внутреннем исследовательском archive при доступном месте сохранять все raw
extractor/NLI payloads с dedup/compression. Public export создаётся отдельной командой с
policy redaction. Нельзя удалять payload после того, как стало известно, что объект FP/FN:
его уже может быть невозможно восстановить без model call.

## 17. Cache subsystem

### 17.1. Принцип

Cache — воспроизводимый научный artifact. Namespace никогда не перезаписывается при смене
prompt/model/schema/revision.

### 17.2. Identity

Ключ должен включать всё, меняющее результат:

```text
component/protocol/schema/prompt/verbalizer/chunking/normalization versions
model logical name + immutable revision
gateway manifest SHA-256
generation parameters
canonical input hashes
evidence/retrieval policy
dtype/device-sensitive policy, если значимо
```

### 17.3. Режимы

```text
read_write
read_only
cache_only
disabled_with_fresh_namespace
```

`cache_only` miss — run integrity error, не скрытый live call. Второй идентичный run должен
делать 0 HTTP/model calls и давать byte-equivalent scientific records после удаления
ожидаемо изменяющихся timestamps/runtime fields.

### 17.4. Shared context reuse

HalluGraph context/query graph cache индексируется по source/context/config, не response.
GraphEval NLI cache — по premise/hypothesis/model policy. Answer extraction cache может
переиспользоваться для duplicate answer texts, сохраняя отдельную provenance-связь каждого
response ID.

## 18. Failure и empty policies

Framework должен поддерживать и логировать counterfactual outcomes:

```text
exclude
impute
fail_open
fail_closed
report_both
```

Рекомендуемая initial primary reporting policy:

1. Основные continuous metrics — на paired complete cases (`both_status_ok`) с явным
   coverage denominator.
2. Отдельно method-specific available-case metrics.
3. Sensitivity bounds: fail-open и fail-closed для каждого метода.
4. Все statuses/failures — самостоятельные показатели качества системы.

Это рекомендация для freeze-конфига, а не повод скрыть unscorable cases.

Обязательные degenerate cases перечислены в разделе 18 artifact spec и должны иметь
unit/integration tests.

## 19. Train-only tuning и freeze

### 19.1. Grouped development

Использовать folds по `source_id`. Сохранять fold manifest и out-of-fold predictions.
Seed и splitter version фиксируются.

### 19.2. GraphEval

Primary tunable parameter — response threshold на raw score. Paper threshold `0.5`
сохраняется как отдельная fixed baseline decision. Изменение threshold не вызывает NLI.

### 19.3. HalluGraph

- Alpha пересчитывается из сохранённых EG/RP без extraction.
- Threshold пересчитывается из H без inference.
- Tau_e/tau_r требуют нового matching либо candidate-lattice replay, но не extraction.
- Все grids, objectives, tie-breakers и selected values сохраняются.

### 19.4. Objective

Primary objective и tie-breaker задаются конфигом до test. Поддержать минимум:

```text
max F1
max balanced accuracy
TPR at fixed FPR
cost-sensitive utility
```

Не выбирать лучший objective после просмотра test.

### 19.5. Calibration

Raw detector score не называть вероятностью без calibration. Поддержать train-only
Platt/isotonic calibration с grouped OOF; сохранять uncalibrated и calibrated результаты.
Brier/ECE для raw score маркировать как score calibration diagnostic либо считать на
calibrated probability.

### 19.6. Threshold manifest

Содержит training sample/fold hashes, gold policy, objective, grid, OOF metrics, selected
parameters, comparator (`>`/`>=`), code hash и freeze timestamp.

## 20. Evaluation

### 20.1. Gold policies

Считать раздельно:

```text
primary_all_labels
exclude_due_to_null
exclude_implicit_true
exclude_both
```

Primary по умолчанию — опубликованная convention: наличие хотя бы одного counted span,
включая `implicit_true` и `due_to_null`. Любое изменение — sensitivity policy.

Quality primary включает все записи и отдельно показывает sensitivity без
`incorrect_refusal`/`truncated`.

### 20.2. Response metrics

Минимум:

```text
AUROC, AUPRC
balanced accuracy, precision, recall, F1, specificity, NPV
confusion matrix
TPR at fixed FPR
Brier, ECE для корректно интерпретируемых scores
coverage/abstention/failure rates
```

Каждая metric row содержит point estimate, 95% CI, n responses, n sources, numerator,
denominator, slice, threshold, gold/failure policy и bootstrap provenance.

### 20.3. Confidence intervals

Cluster bootstrap resamples `source_id` with replacement и включает связанные responses.
Сохраняются replicate-level values и invalid replicate counts. Для малых samples CI и
дискретность явно отмечаются.

### 20.4. Парные сравнения

На идентичных usable responses:

- разность AUROC/AUPRC/F1/etc. с paired cluster bootstrap;
- exact/appropriate McNemar для frozen binary decisions;
- counts discordant pairs;
- standardized score differences только после заранее выбранной transform;
- `both_correct`, `both_wrong`, `hallugraph_only_correct`, `grapheval_only_correct`.

### 20.5. Обязательные срезы

Реализовать весь список раздела 22 artifact spec. Quintile boundaries вычисляются на
заранее выбранной train reference population и переносятся на test. В каждом срезе
показывать `n`, CI и failure coverage; слишком малые клетки не интерпретировать.

## 21. Локализация

### 21.1. Predicted units → response spans

Каждый flagged claim/entity/edge сопоставляется одному или нескольким spans ответа.
Алгоритм версионируется и различает:

```text
exact
aligned_posthoc
ambiguous
unavailable_from_extractor
not_found
```

Нельзя выдумывать offsets. Multiple disjoint spans сохраняются many-to-many.

### 21.2. Метрики

После gold join:

```text
character precision/recall/F1/IoU
span exact match
gold span coverage
flagged unit precision
```

Нужно сохранять промежуточные interval unions/intersections, чтобы метрику можно было
перепроверить.

### 21.3. Attribution

Для каждого FP/FN путь локализации должен различать:

```text
claim не извлечён
claim извлечён, но span не найден
evidence не доступно/обрезано
verifier ошибся
aggregation/threshold изменили unit verdict
gold неоднозначен
```

## 22. Анализы после запуска

Все анализы ниже должны запускаться из sealed archive без detector calls.

### 22.1. Базовые

- overall и per-method metrics;
- paired comparison;
- predefined slices;
- calibration/threshold curves;
- confusion matrices;
- status/failure coverage.

### 22.2. Disagreement analysis

- оба factual / оба hallucinated;
- HalluGraph-only / GraphEval-only hallucinated;
- какой метод корректен после gold join;
- разбиение по task/model/length/graph/claim/error type;
- gallery с полным decision trace;
- согласованные сильные ошибки для re-audit.

### 22.3. Pipeline localization analysis

- extractor coverage и empty/sparse graphs;
- entity normalization/matching failures;
- edge alignment, relation direction и type mismatch;
- NLI neutral/contradiction/overflow;
- evidence availability против correctness;
- extraction health score;
- uncertainty decomposition по стадиям.

### 22.4. Cost/performance

- cold/warm/cache-only;
- latency p50/p95/p99 и throughput;
- calls/tokens/cache hit rate;
- cost per response/source/TP/correct decision/localized span;
- quality-cost и quality-latency curves.

### 22.5. Stability

- score variance/range;
- node/edge/claim Jaccard;
- decision/flag agreement;
- order, batch, concurrency и cache replay invariance;
- hardware tolerance отдельно от exact byte identity.

### 22.6. Counterfactual replay

При достаточных artifacts:

- threshold sweep, ROC/PR;
- HalluGraph alpha sweep;
- tau/type/direction/empty/failure policy replay;
- leave-one-node/edge-out CFI sensitivity;
- minimum sufficient flagged units;
- alternative aggregation policies;
- sensitivity gold/quality policies.

Replay, который требует отсутствующий candidate, должен завершаться `not_replayable` с
причиной, а не тихо вызывать модель.

### 22.7. Oracle/hybrid exploratory

- oracle best method per instance;
- oracle evidence candidate из top-k;
- never/always/fixed/learned/oracle NLI router;
- quality at call/cost budget;
- regret vs oracle;
- coverage-risk/selective accuracy.

Все oracle результаты крупно маркируются как недоступная deployment upper bound.

## 23. Blind audit workflow

Команды должны поддержать:

1. создание audit sample manifest до просмотра predictions;
2. export пакета с input и method trace, но со скрытыми method/gold verdicts согласно phase;
3. независимые annotations;
4. import со schema validation;
5. adjudication;
6. только затем unblind/join;
7. расчёт causal taxonomy и method failure causes.

Сохранить selection probabilities, annotator IDs в redacted форме, confidence, timestamps,
blinding state и adjudication history. Таксономия берётся из раздела 23 artifact spec.

## 24. Reporting

Итоговый `reports/report.md` должен содержать:

- точную идентичность dataset/sample/code/config/models;
- ограничения track и размер подвыборки;
- data/prediction/archive validation statuses;
- overall metrics обоих методов с CI и coverage;
- paired differences и McNemar;
- forest/table обязательных slices;
- calibration и threshold provenance;
- localization metrics;
- failure/empty graph profile;
- cold/warm cost profile;
- stability summary;
- disagreement/error catalog;
- явные confirmatory vs exploratory sections;
- limitations и запрет универсального «кто победил» вывода.

Каждая таблица/plot имеет source table hash, reporting code version, n и CI. Reports можно
перегенерировать без inference, не изменяя scientific tables.

## 25. CLI contract

Предлагаемый интерфейс:

```bash
# Данные
python -m experiments.cli data fetch --revision SHA
python -m experiments.cli data audit --dataset PATH
python -m experiments.cli sample create --config YAML
python -m experiments.cli sample validate --manifest JSON
python -m experiments.cli sample materialize --manifest JSON

# План и запуск
python -m experiments.cli run plan --config YAML
python -m experiments.cli run preflight --config YAML
python -m experiments.cli run execute --config YAML [--resume]
python -m experiments.cli run method --run-id ID --method hallugraph
python -m experiments.cli run method --run-id ID --method grapheval
python -m experiments.cli run seal --run-id ID
python -m experiments.cli run replay --run-id ID --cache-only

# Train-only
python -m experiments.cli tune --run-id TRAIN_ID
python -m experiments.cli freeze --config YAML --threshold-manifest JSON

# Evaluation/analysis
python -m experiments.cli evaluate --run-id TEST_ID --gold-dir PATH
python -m experiments.cli analyze metrics|paired|slices|localization RUN_ID
python -m experiments.cli analyze disagreements|failures|cost|stability RUN_ID
python -m experiments.cli analyze policy-replay RUN_ID --policy YAML
python -m experiments.cli audit create|export|import|adjudicate ...
python -m experiments.cli report build --run-id ID
python -m experiments.cli archive validate --run-id ID
python -m experiments.cli archive export --run-id ID --policy internal|public
```

Все mutating команды по умолчанию показывают resolved paths/run IDs. Нельзя принимать
secret value как CLI argument. `--dry-run` строит plan и preflight, не создавая model calls.

## 26. DataSphere-ready инфраструктура

Облачный этап не должен блокировать локальный framework, но abstractions создаются сразу.

### 26.1. Execution backend

```python
LocalExecutionBackend
DataSphereExecutionBackend
```

Backend отвечает только за запуск, resources, environment и перенос archives. Scientific
runner остаётся тем же.

### 26.2. Data staging

- Полный закреплённый RAGTruth один раз кладётся в shared project storage.
- Job получает dataset checksum и sample manifest, а не скачивает данные заново.
- Job пишет только в `$DS_PROJECT_HOME/runs/$RUN_ID`/выделенный output root.
- Dataset/model ready markers проверяются до model startup.

### 26.3. Gateway

- Использовать только `HALLU_GATEWAY_URL` и `HALLU_GATEWAY_API_KEY` от DataSphere.
- Значение секрета не печатать и не сохранять.
- Перед live run получить authenticated gateway manifest.
- Его canonical SHA-256 входит в run и cache identity.
- 429/5xx/network — bounded backoff до общего wall-time;
- 400/401/403/404 — fail-fast.

### 26.4. Job preflight

До дорогого запуска:

1. code commit/dirty check;
2. dataset/sample checksums;
3. dependency/model revisions;
4. gateway manifest;
5. 2–3 record structured-output probe;
6. artifact schema validation;
7. cache-only replay с 0 calls;
8. disk/RAM/GPU capacity;
9. output directory uniqueness.

### 26.5. Runtime identity

Записывать immutable image digest, Python/dependencies, CUDA/device, DataSphere job ID,
region и resource profile. Image задаётся digest, не плавающим tag.

## 27. Безопасность и redaction

- Не сохранять API keys, bearer tokens, signed URLs и полный environment dump.
- Хранить имя env var и boolean `credential_present`.
- Redactor применяется к exception messages, command line, config и logs до записи.
- Stack trace хранить redacted/hash в публичном archive; internal trace — по policy.
- Public export проходит отдельный privacy audit, поскольку RAGTruth содержит адреса,
  имена и отзывы.
- Detection/artifact code не должен логировать полный prompt через стандартный logger в
  обход content-addressed policy.

## 28. Dependencies и reproducibility

Создать отдельный `requirements.experiments.txt` и lock/constraints artifact. Предпочитать
минимальные зависимости и совместимость Python 3.10–3.12 live runtime.

Ожидаемые категории:

- PyYAML/jsonschema для strict config/schema;
- numpy/pandas/scipy/scikit-learn для analysis;
- pyarrow для Parquet;
- zstandard для payload compression;
- psutil для resource metrics;
- pytest для tests.

Не заставлять offline contract/data tests импортировать torch, transformers, openai или
kg-gen. Heavy imports остаются lazy/optional.

## 29. Тестовая стратегия

### 29.1. Unit

- canonicalization/hash/ID stability;
- raw line preservation;
- RAGTruth schema и offsets;
- Data2txt serializer parity + trace;
- sampling determinism/order independence/group integrity;
- config strictness/freeze hash;
- score/status invariants;
- metrics на hand-computed fixtures;
- cluster bootstrap;
- span interval math;
- atomic writer/payload dedup/redaction.

### 29.2. Contract

- один fake `DetectionInput` проходит оба detectors;
- gold denylist тестируется рекурсивно;
- HalluGraph adapter parity с `run.build_rows`;
- GraphEval answer extractor действительно получает только response;
- одинаковые response/source IDs и score direction;
- capabilities соответствуют фактическим artifacts.

### 29.3. Integration offline

На synthetic RAGTruth fixture:

```text
fetch/import fixture → audit → sample → materialize → both fake detectors
→ seal → gold join → tune/evaluate → report → archive validate
```

Тест должен работать без сети/torch/openai.

### 29.4. Resilience

- interrupt/resume после каждой стадии;
- оборванная JSONL строка;
- corrupt cache/payload/checksum;
- duplicate response text с разными IDs;
- cache-only miss;
- per-instance failure;
- run-level auth/config failure;
- concurrent writers;
- повторный finalize/seal.

### 29.5. Leakage

- detector input serialization не содержит gold keys;
- prediction worker не открывает gold path;
- gold join запрещён до seal;
- train/test source overlap validator;
- tuning отклоняет test split;
- audit unblinding state machine.

### 29.6. Reproducibility

- input order invariance;
- cache-only scientific equivalence;
- batch/concurrency invariance для deterministic fake;
- report regeneration identity;
- checksum verification всего archive.

### 29.7. Live gates

1. real 2–3 record gateway/HHEM probe;
2. warm cache-only replay с 0 calls;
3. маленький multi-task probe;
4. development subset;
5. только затем frozen test sample.

## 30. План реализации

Каждый этап заканчивается отдельным логическим commit и зелёными tests. Не объединять
instrumentation двух методов, statistics и DataSphere в один непроверяемый change.

### Этап 0. Baseline freeze

Deliverables:

- зафиксировать base commit/branch;
- запустить существующие offline tests;
- сохранить baseline test counts;
- задокументировать текущие method capabilities и gaps;
- проверить clean worktree перед изменениями.

Acceptance:

- GraphEval tests и HalluGraph adapter parity зелёные;
- никакая scientific formula не изменена.

### Этап 1. Нейтральный detector contract

Deliverables:

- `detector_contracts`;
- compatibility re-exports;
- `DetectorProtocol`;
- symmetric `GraphEvalAdapter`/registry;
- contract/leakage tests.

Acceptance:

- старые imports продолжают работать;
- оба метода вызываются одинаково;
- adapter parity не изменился.

### Этап 2. RAGTruth raw store и audit

Deliverables:

- pinned downloader/importer;
- manifests/checksums;
- raw/canonical/derived records;
- полный schema/data audit;
- Data2txt trace и QA passage provenance.

Acceptance:

- повторный fetch проверяет checksum;
- raw bytes не меняются;
- все опубликованные IDs joinятся либо проблема явно отчётна;
- invalid offsets видны.

### Этап 3. Sampling/materialization/gold isolation

Deliverables:

- generic source-level sampler;
- immutable sample manifest;
- no-gold/gold split;
- purpose presets;
- manifest replay.

Acceptance:

- deterministic при перестановке входных строк;
- нет source leakage;
- test manifest не использует gold/predictions;
- prediction input проходит recursive denylist.

### Этап 4. Artifact foundation

Deliverables:

- schemas, sink, atomic JSONL/Parquet writers;
- content-addressed compressed payload store;
- stable IDs, lineage, checksums, redaction;
- run state machine и seal.

Acceptance:

- crash/resume/corruption tests;
- каждый ref разрешим;
- gold join невозможен до seal.

### Этап 5. Generic runner на fake backends

Deliverables:

- config/planner/preflight;
- detector registry;
- paired execution/resume;
- stage/failure/usage logging;
- end-to-end offline CLI.

Acceptance:

- полный offline fixture archive;
- exact paired ID coverage;
- repeat/cache-only scientific equivalence.

### Этап 6. GraphEval instrumentation

Deliverables:

- extraction/parse/repair trace;
- claim/provenance/span alignment;
- NLI input/output/context access;
- aggregation decision trace;
- full cache identity.

Acceptance:

- исходный GraphEval score byte/numeric parity;
- extractor-sees-answer-only test;
- HHEM label direction test;
- empty/partial/failure cases.

### Этап 7. HalluGraph instrumentation

Deliverables:

- raw/normalized graphs;
- candidate lattice и decisions;
- EG/RP/CFI aggregation trace;
- cache/context reuse/resource metrics.

Acceptance:

- parity с `run.build_rows` на всех существующих modes;
- candidate trace объясняет каждый match/rejection;
- tau/alpha replay корректен там, где хватает lattice.

### Этап 8. Tuning/freeze/policy replay

Deliverables:

- source-grouped folds/OOF;
- threshold/alpha/tau tuning;
- calibration;
- frozen threshold manifest;
- empty/failure/gold/quality policy replay.

Acceptance:

- test input отклоняется tuner-ом;
- hand-computed fixtures совпадают;
- downstream replay не вызывает модели.

### Этап 9. Evaluation/statistics

Deliverables:

- response metrics;
- source-cluster bootstrap;
- paired differences/McNemar;
- predefined slices;
- calibration tables;
- localization metrics.

Acceptance:

- каждая metric row содержит n/CI/policies/provenance;
- tiny/degenerate slices возвращают controlled undefined, не ложный 0;
- paired denominator проверяется.

### Этап 10. Analysis/audit/reporting

Deliverables:

- disagreements/failures/cost/stability;
- blind audit workflow;
- report/tables/plots;
- confirmatory/exploratory separation;
- archive validator/export policy.

Acceptance:

- report полностью строится из sealed archive;
- FP/FN открывается до machine trace;
- public export не содержит secrets.

### Этап 11. Controlled tracks и typed/hybrid hooks

Deliverables:

- shared graph IR;
- shared answer/all graph runners;
- B0–B4 type ablations;
- shadow NLI/router feature tables;
- oracle upper bounds.

Acceptance:

- shared artifact hashes действительно одинаковы между methods;
- faithful и controlled results не смешиваются;
- router leakage tests.

### Этап 12. DataSphere backend

Deliverables:

- job template/renderer/validator;
- staging checks;
- gateway manifest integration;
- probe → cache-only → pilot gates;
- archive download/validation docs.

Acceptance:

- image digest и model revisions pinned;
- secrets отсутствуют в files/logs;
- второй run делает 0 live calls;
- downloaded archive проходит локальный validator.

## 31. Итоговые acceptance criteria

Framework не готов, пока не выполнено всё следующее:

1. Полная версия и checksums RAGTruth закреплены.
2. Подвыборка воспроизводится из manifest.
3. Detector input физически не содержит gold.
4. Оба метода получают идентичные `(context, query, response)` данного track.
5. Парные raw predictions существуют для ожидаемых IDs или имеют явный failure status.
6. HalluGraph EG/RP/CFI и GraphEval claim NLI scores сохранены.
7. Каждый score объясняется graph/claim/candidate/decision trace.
8. Empty/failure policies заморожены и имеют sensitivity report.
9. Threshold ссылается на train-only OOF artifact.
10. Flagged units связаны с response spans и evidence candidates.
11. Cost/latency/cache и stability profiles присутствуют.
12. Metrics имеют n, CI, slice, gold/failure policies и bootstrap unit.
13. Disagreement catalog и blind audit workflow доступны.
14. Test не использован в tuning.
15. Archive проходит leakage, schema, lineage и checksum validation.
16. Cache-only replay не делает network/model calls.
17. Existing method parity tests не регрессировали.
18. Track name соответствует реально использованным components.

## 32. Решения, которые нужно заморозить перед первым live test

Агент должен подготовить конфиг и явно запросить/зафиксировать решение владельца по:

- exact RAGTruth commit SHA и checksums;
- размеру и strata development/test samples;
- primary исполнимому track (`kggen_untyped_adaptation` рекомендуется первым);
- GraphEval extractor mode (`paper_prompt`/`structured_json`);
- exact HHEM Hugging Face revision;
- HalluGraph relation mode;
- alpha/tau grids и tuning objective;
- frozen response thresholds;
- full-context/chunking/evidence policy;
- empty/failure primary policy;
- primary gold/quality policies;
- bootstrap replicates/seed и fixed FPR;
- audit sample size/strata;
- retention/publication policy;
- local или DataSphere execution profile.

До этих решений разрешены только offline framework tests и development probes. Test run
не запускать.

## 33. Правила работы следующего агента

1. Сначала прочитать этот документ и `docs/graph_eval_integration.md` полностью.
2. Проверить branch/status и сохранить пользовательские изменения.
3. Не переписывать scientific formulas ради удобства framework.
4. Не смешивать refactor с изменением метрик/порогов.
5. Использовать offline fake-backends до live gates.
6. Не обращаться к test при реализации/tuning.
7. Не выводить и не запрашивать секреты.
8. Любое намеренное отклонение записывать в отдельный deviations log с причиной и
   влиянием на scientific validity.
9. После каждого этапа запускать узкие и затем общие regression tests.
10. Не считать работу законченной только потому, что CLI выдаёт один score: главным
    deliverable является валидируемый, воспроизводимый и анализируемый archive.
