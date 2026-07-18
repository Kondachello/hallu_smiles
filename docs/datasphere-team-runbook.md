# DataSphere team runbook: immutable local-LLM runtime

Этот runbook — технический контракт запуска `new-metrics` в существующем
DataSphere Project. «Локальная LLM» здесь означает Llama, обслуживаемую vLLM на
`127.0.0.1` **внутри удалённой GPU Job**. На рабочем Mac не устанавливаются ни
веса, ни CUDA runtime, ни inference server.

## Архитектура

```text
локальный checkout
  └─ commit + push → GitHub-hosted remote build → immutable GHCR digest
                                                        │
                                  DataSphere CLI submit ─┤
existing DataSphere Project                     │
  ├─ public OCI image@sha256 ───────────────────┤
  │    ├─ /opt/hallu/server  (vLLM + XGrammar + CUDA PyTorch)
  │    ├─ /opt/hallu/client  (KGGen/DSPy/scoring + CPU PyTorch)
  │    ├─ /opt/hallu/models/all-MiniLM-L6-v2
  │    └─ runtime manifest + two pip freezes
  │
  ├─ read-only Project disk
  │    └─ hallu_smiles/shared/{models,ragtruth}
  │
  └─ per-Job writable storage
       └─ probes, caches, reports, logs, archive
```

Три идентификатора образуют воспроизводимый runtime:

1. полный 40-символьный Git SHA;
2. immutable OCI manifest digest (либо optional DataSphere resource ID);
3. exact model revision из shared model manifest.

Они записываются в `runtime-manifest.json`, `run_metadata.json` и cache
fingerprints. Job не считается воспроизводимой, если эти значения расходятся.

## Неподвижные ограничения

- Project ID: `bt1i64odluitglbaj5st`.
- GPU profile: один `g1.1`, V100 32 GB, FP16, `max-model-len=8192`.
- Один GPU Job одновременно.
- Модель и RAGTruth доступны только из attached Project disk; GPU Job не
  скачивает их и не пишет в shared storage.
- `HF_TOKEN` отсутствует в Job. Runner дополнительно делает `unset HF_TOKEN`.
- В Job запрещены `pip install`, сборка environment и model download.
- Клиент запускается с `CUDA_VISIBLE_DEVICES=""`; GPU принадлежит только vLLM.
- KGGen extraction и official LLM-clustering выполняются последовательно,
  concurrency = 1. Full pilot не ограничивает cluster items.
- Official `KGGen.cluster` получает штатный `context`: фиксированный strict-equivalence
  policy и только тот текст, из которого сейчас строится соответствующий `G_c`, `G_q`
  или `G_a`.
- Strict/support используют один manifest, KG cache и verifier cache.
- Tuning выполняется только на 16 train; test из четырёх строк оценивается один
  раз после заморозки параметров.

## Shared storage contract

Обычная Job читает:

```text
$DS_PROJECT_HOME/hallu_smiles/shared/
  models/
    active-model.json
    <content-addressed-model-directory>/
      ...weights...
      .ready.json
  ragtruth/
    source_info.jsonl
    response.jsonl
```

Staging нужен только если ready marker отсутствует или сознательно меняется HF
revision. Его выполняют в CPU Jupyter, где `HF_TOKEN` существует временно:

```bash
cd /home/jupyter/project/hallu_smiles
export DS_SHARED_ROOT="$PWD/shared"
python scripts/stage_datasphere_shared_assets.py \
  --shared-root "$DS_SHARED_ROOT" \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct
```

`DS_PROJECT_HOME` — переменная Job; в Jupyter используйте путь проекта. После
staging освободите Jupyter VM. Не удаляйте и не изменяйте existing shared
assets из Job.

## Immutable Docker runtime

Источник: `datasphere/docker/Dockerfile.template`. Он строится удалённо и не
копирует локальный checkout, 8B модель, RAGTruth или Project storage.

Server environment `/opt/hallu/server` включает exact pins из
`datasphere/docker/server.requirements.txt`, в том числе:

```text
vllm==0.8.5.post1+cu118
torch==2.6.0+cu118
torchvision==0.21.0+cu118
torchaudio==2.6.0+cu118
xformers==0.0.29.post2 (wheel from the cu118 index)
transformers==4.51.3
xgrammar==0.1.18
```

GPU packages are pinned to direct CUDA 11.8 wheel URLs. Это обязательная
совместимость с подтверждённым `g1.1`/V100 driver API 12.2: дефолтный CUDA
12.4 wheel для этой версии vLLM на том же профиле уже завершался ошибкой
`driver too old`. Docker build, CPU preflight и GPU preflight независимо
проверяют exact `torch` build; несовпадение останавливает запуск до модели.

Client environment `/opt/hallu/client` включает exact KGGen/DSPy/LiteLLM/
Pydantic/scientific pins, CPU-only `torch==2.6.0`, `jsonschema` и
`sentence-transformers`. Exact MiniLM revision скачивается только во время
remote image build и после сборки используется офлайн.

### Удалённая CLI-only сборка

Сначала commit должен быть опубликован:

```bash
git switch new-metrics
git push origin new-metrics
COMMIT=$(git rev-parse HEAD)
git fetch origin refs/heads/new-metrics
git merge-base --is-ancestor "$COMMIT" FETCH_HEAD
```

Push автоматически запускает `.github/workflows/datasphere-runtime-image.yml`.
GitHub-hosted `ubuntu-22.04` runner рендерит Dockerfile по `GITHUB_SHA`, собирает
`linux/amd64`, публикует commit tag в public GHCR, затем выходит из registry и
проверяет anonymous pull exact digest. На Mac не нужен Docker и не сохраняются
image layers. Статус проверяется через CLI:

```bash
gh run list --repo Kondachello/hallu_smiles --branch new-metrics \
  --workflow datasphere-runtime-image.yml --limit 1
```

Разрешение commit tag в exact digest тоже не требует Docker daemon:

```bash
.venv-datasphere/bin/python scripts/resolve_datasphere_runtime_image.py \
  --commit "$COMMIT"
```

Submit helper выполняет это разрешение сам и записывает в YAML только
`ghcr.io/...@sha256:...`; mutable tags fail-closed запрещены. Manifest внутри
image содержит source commit, SHA двух freezes, embedding revision/path и общий
runtime fingerprint. CPU preflight отклоняет image, построенный из другого SHA.
Если команда уже располагает DataSphere Project Docker resource, его можно
явно передать через `--docker-image-id b...`; это совместимый optional path.

Практическое следствие: изменение кода, который входит в runtime contract,
requirements или Dockerfile требует commit → push → успешной remote image build →
нового CPU preflight. Никогда не проверяйте новую Job старым digest «для
экономии одного шага».

## Локальная настройка DataSphere CLI

В checkout уже находятся `.venv-datasphere/bin/datasphere` и `.tools/yc/yc`.
Wrapper сам добавляет `.tools/yc` в `PATH`, поэтому повторная установка и
повторный `yc init` не нужны. Проверка:

```bash
PATH="$PWD/.tools/yc:$PATH" \
.venv-datasphere/bin/datasphere --profile default project get \
  --id bt1i64odluitglbaj5st
```

Для federated account используйте plain `yc init`. Отсутствие списка clouds не
даёт права создавать новый cloud/Project и не мешает доступу к выданному
Project. Секреты и вывод авторизации не публикуйте.

## Единственный поддерживаемый submit path

Не отправляйте template YAML напрямую и не вызывайте `job execute` вручную.
Используйте `scripts/submit_datasphere_job.sh`; он:

1. проверяет аргументы и разрешает exact commit в immutable OCI digest;
2. убеждается, что commit входит в `origin/<branch>`;
3. рендерит конкретный YAML;
4. валидирует `bash -lc`, attached project disk, `env.docker`, отсутствие
   `env.python`, `pip install`, model download и неразрешённых shell variables;
5. проверяет существующий Project;
6. для GPU Job валидирует tar предыдущего gate с exact SHA/image/model;
7. fail-closed проверяет, что другая HalluGraph GPU Job не активна;
8. создаёт Job асинхронно и сохраняет JSON с `job_id`/`operation_id`.

Общая команда:

```bash
PROJECT_ID=bt1i64odluitglbaj5st
BRANCH=new-metrics

bash scripts/submit_datasphere_job.sh \
  --kind <preflight|cluster-probe-g1|qa-pilot-g1> \
  --project-id "$PROJECT_ID" \
  --branch "$BRANCH" \
  --commit "$COMMIT" \
  --run-id <unique-lowercase-run-id>
```

Для `cluster-probe-g1` дополнительно обязателен `--gate-artifact` с tar
успешного preflight; для `qa-pilot-g1` — tar чистого 3-QA probe. Для CPU
preflight этот аргумент запрещён.

## Gate 1: CPU preflight

```bash
RUN_ID=preflight-$(git rev-parse --short HEAD)-$(date +%Y%m%d)
bash scripts/submit_datasphere_job.sh \
  --kind preflight --project-id "$PROJECT_ID" --branch "$BRANCH" \
  --commit "$COMMIT" --run-id "$RUN_ID"
```

`c1.4` Job использует тот же immutable image digest, но не загружает 8B weights и не
резервирует GPU. Extended `working-storage` для неё намеренно не запрашивается:
небольшие отчёты помещаются в системную рабочую директорию, поэтому отдельные
100 ГБ SSD не тарифицируются. Она проверяет:

- shared model ready marker, revision, size и RAGTruth;
- импорты и exact versions отдельно в server/client environments;
- компиляцию XGrammar для closed relation/verifier schemas и всех реальных
  input-dependent KGGen clustering schemas;
- local JSON Schema validation;
- offline CPU S-BERT encode с exact snapshot;
- соответствие source commit и embedding identity runtime manifest.

Разрешающий результат — terminal `SUCCESS`, `preflight.json.status=ready`,
`gate_metadata.json.state=completed` и
`runtime-dependencies.json.status=ready`. Archive также сохраняет schemas,
runtime manifest и оба freeze. При любой ошибке GPU Job запрещена.

## Structured-output contract

DSPy/KGGen передаёт серверу точную закрытую schema нативно:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "...",
      "strict": true,
      "schema": {"type": "object", "additionalProperties": false}
    }
  }
}
```

Backend vLLM — XGrammar. Ответ допустим только если:

- `finish_reason == "stop"`;
- существует ровно один choice;
- JSON parses и проходит исходную schema;
- relation extraction имеет корневой `relations` array;
- verifier имеет ровно один enum verdict: `entailed`, `contradicted` или
  `unknown`.

Schema/parse validation error считается детерминированной и завершается
fail-fast без повторного LLM-вызова или parser repair. Повторяются только
ограниченные по числу transient timeout/connection/5xx/429 ошибки. Нельзя добавлять fallback на unconstrained JSON, оборачивать
bare triple или отключать official LLM-clustering.

Для pinned KGGen `0.4.0` действует versioned compatibility contract
`strict-response-format-v4-xgrammar-runtime-input-contracts`. Он не заменяет
`KGGen.cluster` и не меняет его LLM control flow, но делает заявленные
динамические ограничения исполнимыми в XGrammar и в независимом parser check:

- `Relation.subject/object` принадлежат текущему `entities`;
- `ExtractCluster` и `ValidateCluster` возвращают только текущие кандидаты;
- `ChooseRepresentative` выбирает member текущего validated cluster;
- `CheckExistingClusters` возвращает ровно по одному existing representative
  либо `null` на каждый item.

Последнее правило для representative сознательно усиливает upstream «ideally»
до `MUST`: иначе одинаковый свободный label для двух кластеров перезаписывает
один из них в штатном словаре KGGen. Никакого post-hoc исправления ответа или
графа нет. Clustering context дополнительно формулирует строгую эквивалентность:
тематическая связь, co-occurrence и общий endpoint не являются синонимией.

## Gate 2: exact probes и три QA

```bash
RUN_ID=cluster-probe-$(git rev-parse --short HEAD)-$(date +%Y%m%d)
PREFLIGHT_ARCHIVE=<PATH_TO_DOWNLOADED_preflight-*.tar.gz>
bash scripts/submit_datasphere_job.sh \
  --kind cluster-probe-g1 --project-id "$PROJECT_ID" --branch "$BRANCH" \
  --commit "$COMMIT" \
  --gate-artifact "$PREFLIGHT_ARCHIVE" --run-id "$RUN_ID"
```

Runner на одном V100 делает:

1. shared-assets и GPU runtime checks;
2. vLLM `/health` и короткий completion;
3. точный real KGGen relation prompt/schema один раз;
4. KGGen typed extraction + official LLM-clustering;
5. verifier closed-schema probe;
6. first selected real RAGTruth reference graph;
7. extraction трёх fixed manifest rows.

Gate пройден только при terminal `SUCCESS`, всех probe reports `ready`, пустом
`strict/failed_extractions.jsonl`, завершённых reference/answer graph pairs и
валидном `cache/cluster-audit.jsonl` с полным покрытием cluster mappings, а также
наличии GPU activity во время inference. Timeout и 180-second idle watchdog
ограничивают цену сбоя. AUC здесь не считается.

## Gate 3: full 20-QA pilot

```bash
RUN_ID=qa-pilot-$(git rev-parse --short HEAD)-$(date +%Y%m%d)
CLUSTER_ARCHIVE=<PATH_TO_DOWNLOADED_cluster-probe-*.tar.gz>
bash scripts/submit_datasphere_job.sh \
  --kind qa-pilot-g1 --project-id "$PROJECT_ID" --branch "$BRANCH" \
  --commit "$COMMIT" \
  --gate-artifact "$CLUSTER_ARCHIVE" --run-id "$RUN_ID"
```

Порядок неизменяем:

1. те же exact probes;
2. strict extraction создаёт 20-QA manifest и KG cache;
3. strict scoring/tuning/evaluation с `--kg-cache-only`;
4. support scoring с тем же manifest и `--kg-cache-only`; SHA graph cache до
   и после support обязаны совпасть, verifier cache при этом заполняется live;
5. comparison;
6. stop vLLM;
7. strict и support replay с `--cache-only`;
8. byte comparison metrics и SHA-256 cache tree.

Cache keys включают model ID/revision, runtime fingerprint, structured-output
transport/backend/schema contract, `cluster_context_mode`, extraction parameters и content. Fake/live
namespaces разделены. `--cache-only` запрещает live inference; cache miss —
немедленная ошибка. S-BERT тоже работает только с embedded local path.

Full Job успешна, только если `failed_extractions.jsonl` пуст, replay сделал
ноль live calls, strict/support `metrics.csv` byte-identical соответствующим
live runs, а `cache-before.sha256` равен `cache-after.sha256`.

## Runtime budget and liveness

- `cluster-probe-g1`: максимум 3600 секунд, idle extraction watchdog 180 сек.
- `qa-pilot-g1`: максимум 10800 секунд плюс до 60 секунд graceful shutdown,
  idle extraction watchdog 600 сек.
- `nvidia-smi` пишет `gpu.csv` каждые 10 секунд.
- Watchdog действует только во время live KG extraction. CPU scoring, tuning,
  reporting и cache-only replay не должны отменяться из-за 0% GPU.
- Exit trap останавливает vLLM и пытается упаковать артефакты при success,
  error, cancel или timeout.

## Наблюдение, отмена и результаты

```bash
.venv-datasphere/bin/datasphere --profile default project job get \
  --id <JOB_ID> --format json
```

`PREPARING` теперь означает запуск immutable container и mount storage, а не
долгую установку Python packages. В `EXECUTING` смотрите Launch history и
tails `pilot.stdout.log`, `pilot.stderr.log`, `vllm.log`, `gpu.csv`. На macOS
`job attach` может показывать gRPC warnings; `job get` и UI надёжнее.

Если нужен cancel одной Job:

```bash
.venv-datasphere/bin/datasphere --profile default project job cancel \
  --id <JOB_ID> --graceful
```

После terminal status:

```bash
mkdir -p "outputs/datasphere-results/$RUN_ID"
.venv-datasphere/bin/datasphere --profile default project job download-files \
  --id <JOB_ID> --with-logs --with-diagnostics \
  --output-dir "outputs/datasphere-results/$RUN_ID"
```

Обязательные QA artifacts:

- runtime/model identity: `runtime-manifest.json`, `server.freeze.txt`,
  `client.freeze.txt`, `shared-assets-preflight.json`, `gpu-runtime.json`,
  `runtime_config.yaml`, `run_metadata.json`;
- protocol gates: `vllm-response-format-probe.json`, `kggen-probe.json`,
  `verifier-probe.json`, `qa-reference-probe.json`;
- experiment: `qa_pilot_manifest.json`, `strict/`, `support/`,
  `comparison.json`;
- cache proof: `cache-replay/`, `cache-before.sha256`,
  `cache-after.sha256`;
- diagnostics: `vllm.log`, `gpu.csv`, `pilot.stdout.log`,
  `pilot.stderr.log`.

## Failure policy

| Failure | Required action |
|---|---|
| Docker build fails | Исправить pins/build, commit+push, дождаться нового digest. Не чинить packages внутри Job. |
| Preflight fails | Не выделять V100. Исправить runtime и повторить только CPU preflight. |
| Exact relation schema fails | Не запускать three-QA/full pilot и не делать parser repair. |
| KGGen cluster/verifier/real-reference probe fails | Сохранить diagnostics, исправить точную границу, повторить bounded probe. |
| Three-QA extraction incomplete | Full pilot запрещён. |
| Full live extraction incomplete | AUC не интерпретировать; Job должна быть error. |
| Cache-only replay fails | Archive не доказан воспроизводимым; результат не принимать. |
| 0% GPU только в CPU phase | Нормально; idle watchdog ограничен extraction phase. |

## Pre-submit safety checks

`scripts/validate_datasphere_job.py` требует:

- `cmd` в `bash -lc` и валидный shell syntax;
- `flags: [attach-project-disk]`;
- `env.docker.image` с immutable OCI digest (или optional project resource ID) и отсутствие `env.python`;
- запись immutable image identity в runtime metadata;
- отсутствие runtime `pip install` и model downloads;
- отсутствие shell forms, которые DataSphere CLI ошибочно интерпретирует как
  undeclared YAML variables.

Если helper остановился до `job execute`, Job не создана и units не потрачены.

## Legacy incident record

Предыдущий manual runtime (`env.python.type: manual`, vLLM `0.6.3.post1`,
Outlines `0.0.46`, LM Format Enforcer `0.10.6`, Transformers `4.45.2`,
`pyairports` shim и boolean-schema patch) больше не является поддерживаемым
способом запуска. На упрощённой schema он мог выглядеть исправным, но на
реальном KGGen relation prompt нарушал корневой контракт и возвращал bare
`{"subject", "predicate", "object"}`. Эти сведения сохранены только для
диагностики старых архивов; не переносите старые shims, `guided_json` transport
или manual environment в новый Docker runtime.

## Командные правила

- Один Job = один pushed SHA = один immutable Docker ID = один `RUN_ID` = один
  archive.
- Gates выполняются строго: CPU preflight → exact/three-QA probe → full 20 QA
  → cache-only proof.
- Не запускайте больше одной GPU Job.
- Не изменяйте shared assets без согласования и нового preflight.
- Не принимайте отчёт без model/runtime identity, пустого failures file и
  успешного cache-only replay.
