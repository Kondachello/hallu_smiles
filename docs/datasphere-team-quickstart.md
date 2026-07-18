# DataSphere: памятка команде — безопасный запуск `new-metrics`

Эта памятка описывает актуальный путь от локальной ветки до 20-QA отчёта.
Ноутбук только рендерит и отправляет DataSphere Jobs. Llama 3.1 8B работает
локально относительно удалённой GPU Job — через vLLM на `127.0.0.1:8000` — и
не устанавливается на ноутбук.

Полная техническая справка: [datasphere-team-runbook.md](datasphere-team-runbook.md).

## Что уже подготовлено и что нельзя менять

- Модель `meta-llama/Meta-Llama-3.1-8B-Instruct` и RAGTruth уже находятся в
  Project storage под `$DS_PROJECT_HOME/hallu_smiles/shared`. Job читает их
  read-only и ничего туда не записывает.
- `HF_TOKEN` нужен только для одноразового staging в CPU Jupyter. Он не нужен
  и не передаётся ни в CPU preflight, ни в GPU Jobs.
- Среда Job — public OCI image, неизменяемо закреплённый digest’ом. В Job запрещены
  `pip install`, скачивание модели и сборка окружения.
- Docker image разделяет несовместимые роли:
  `/opt/hallu/server` содержит vLLM `0.8.5.post1+cu118`,
  PyTorch `2.6.0+cu118`, Transformers и
  XGrammar; `/opt/hallu/client` содержит KGGen, DSPy, LiteLLM, verifier,
  scoring и CPU-only PyTorch.
- CUDA 11.8 выбрана намеренно: сохранённый runtime probe на реальном `g1.1`
  подтвердил V100 и driver API 12.2; CUDA 12.4 на этом узле уже падала до
  запуска модели. Оба preflight-а проверяют exact CUDA build fail-closed.
- S-BERT `sentence-transformers/all-MiniLM-L6-v2` с точной revision встроен в
  image по пути `/opt/hallu/models/all-MiniLM-L6-v2`. Клиент загружает его с
  `local_files_only=True`, `HF_HUB_OFFLINE=1` и `CUDA_VISIBLE_DEVICES=""`.
- KGGen LLM-clustering остаётся включённой. Full 20-QA pilot не ограничивает
  число элементов кластеризации и не подменяет её эвристикой.
- Штатный `KGGen.cluster(..., context=...)` получает только текущий извлекаемый
  текст; `G_c`, `G_q` и `G_a` не смешиваются. Mappings сохраняются в
  `cache/cluster-audit.jsonl`.
- Structured output передаётся только нативным OpenAI-compatible полем
  `response_format` типа `json_schema`; сервер использует XGrammar. Bare
  relation object не оборачивается постфактум в `{"relations": [...]}`.

## Обязательная последовательность gates

```text
push полного commit SHA
        │
        ├── remote build + public immutable OCI digest
        │
        └── CPU preflight (c1.4)
                  │ SUCCESS
                  └── exact relation/verifier/real-reference probes
                              + bounded extraction первых 3 QA (g1.1)
                                      │ SUCCESS, failed_extractions пуст
                                      └── full 20 QA (g1.1)
                                              └── server stopped
                                                  cache-only strict/support replay
```

Нельзя переходить через неуспешный gate. `cluster-probe-g1` не считает AUC и
не является экспериментальным результатом.

## 0. Настроить CLI на ноутбуке

Из корня репозитория:

```bash
python3.12 -m venv .venv-datasphere
.venv-datasphere/bin/python -m pip install --upgrade pip datasphere
# Установить Yandex Cloud CLI официальным способом, затем:
yc init
```

Для federated аккаунта используйте обычный `yc init`, без `--username`. Если
CLI не показывает доступных clouds, не создавайте новый cloud или Project.
Проверяйте доступ к уже существующему проекту:

```bash
.venv-datasphere/bin/datasphere --profile default project get --id <PROJECT_ID>
```

Не сохраняйте OAuth token, `HF_TOKEN` или другие секреты в Git, YAML, логи и
скриншоты.

## 1. Зафиксировать и опубликовать код

Job клонирует публичный репозиторий по полному SHA. Незакоммиченный или только
локальный код никогда не попадёт в запуск.

```bash
git switch new-metrics
git status --short
git add <только-нужные-файлы>
git commit -m "Describe the DataSphere runtime change"
git push origin new-metrics
COMMIT=$(git rev-parse HEAD)
git merge-base --is-ancestor "$COMMIT" origin/new-metrics
```

Не используйте сокращённый SHA в `--commit`.

## 2. Дождаться immutable remote runtime

Push в `new-metrics` запускает `.github/workflows/datasphere-runtime-image.yml`.
GitHub-hosted Linux runner рендерит Dockerfile по exact SHA, собирает `linux/amd64`
image и публикует его в public GHCR. На ноутбук не скачиваются ни Docker layers,
ни модель. Дождитесь зелёного workflow:

```bash
gh run list --repo Kondachello/hallu_smiles --branch new-metrics \
  --workflow datasphere-runtime-image.yml --limit 1
```

Проверить и получить exact digest можно без Docker daemon:

```bash
.venv-datasphere/bin/python scripts/resolve_datasphere_runtime_image.py \
  --commit "$COMMIT"
```

Image содержит `/opt/hallu/runtime-manifest.json`, `server.freeze.txt`,
`client.freeze.txt` и offline S-BERT snapshot. CPU preflight потребует, чтобы
`runtime-manifest.json.source_commit` строго совпал с `--commit`. Submit helper
сам разрешает commit tag в `image@sha256:...` и отказывается от mutable tag.
Опционально по-прежнему можно передать уже созданный DataSphere Project resource
через `--docker-image-id b...`, но основной CLI-only path не требует UI или ID.

## 3. CPU preflight

```bash
PROJECT_ID=bt1i64odluitglbaj5st
BRANCH=new-metrics
RUN_ID=preflight-$(git rev-parse --short HEAD)-$(date +%Y%m%d)

bash scripts/submit_datasphere_job.sh \
  --kind preflight \
  --project-id "$PROJECT_ID" \
  --branch "$BRANCH" \
  --commit "$COMMIT" \
  --run-id "$RUN_ID"
```

Helper сам добавляет `.tools/yc` в `PATH`, использует `.venv-datasphere`,
проверяет, что SHA содержится в `origin/new-metrics`, разрешает immutable GHCR
digest, затем
рендерит и валидирует YAML, проверяет существующий Project и только после этого
вызывает `datasphere project job execute --async`. Он не создаёт Project,
секреты или Docker resource.

Preflight успешен только при terminal `SUCCESS` и наличии в архиве:

- `preflight.json` со `status: ready`, точной model revision и ненулевым
  `model_bytes_checked`;
- `runtime-dependencies.json` со `status: ready`;
- `preflight-schemas.json`, успешно скомпилированного XGrammar для relation,
  verifier и clustering enum schemas;
- `runtime-manifest.json`, `server.freeze.txt`, `client.freeze.txt`;
- `gate_metadata.json`, связывающий exact source commit, Docker ID, model
  revision и runtime fingerprint;
- успешного offline CPU encode встроенным S-BERT.

## 4. Bounded three-QA probe

Только после успешного CPU preflight и скачивания его tar:

```bash
RUN_ID=cluster-probe-$(git rev-parse --short HEAD)-$(date +%Y%m%d)
PREFLIGHT_ARCHIVE=<PATH_TO_DOWNLOADED_preflight-*.tar.gz>

bash scripts/submit_datasphere_job.sh \
  --kind cluster-probe-g1 \
  --project-id "$PROJECT_ID" \
  --branch "$BRANCH" \
  --commit "$COMMIT" \
  --gate-artifact "$PREFLIGHT_ARCHIVE" \
  --run-id "$RUN_ID"
```

Перед extraction трёх фиксированных QA runner обязан по порядку пройти:

1. `/health` и короткий completion smoke-check.
2. Точный реальный KGGen relation prompt/schema один раз через native
   `response_format`; ответ имеет корень `{"relations": [...]}` и проходит
   локальную JSON Schema validation.
3. KGGen/DSPy typed extraction и официальную LLM-clustering.
4. Закрытую verifier schema с единственным verdict.
5. Первый выбранный RAGTruth reference graph.
6. Полную extraction reference/answer pairs для трёх manifest rows.

Проверьте `vllm-response-format-probe.json`, `kggen-probe.json`,
`verifier-probe.json`, `qa-reference-probe.json`, `run_metadata.json` и
`cache/cluster-audit.jsonl`, а также `strict/failed_extractions.jsonl`.
Последний файл должен быть пустым, а cluster audit — проходить structural checks. Пики GPU
utilisation подтверждают работу inference; нулевой GPU во время CPU preflight
или scoring сам по себе не является зависанием.

## 5. Full 20-QA pilot

Только после чистого three-QA probe и скачивания его tar:

```bash
RUN_ID=qa-pilot-$(git rev-parse --short HEAD)-$(date +%Y%m%d)
CLUSTER_ARCHIVE=<PATH_TO_DOWNLOADED_cluster-probe-*.tar.gz>

bash scripts/submit_datasphere_job.sh \
  --kind qa-pilot-g1 \
  --project-id "$PROJECT_ID" \
  --branch "$BRANCH" \
  --commit "$COMMIT" \
  --gate-artifact "$CLUSTER_ARCHIVE" \
  --run-id "$RUN_ID"
```

Один `g1.1` Job запускает один FP16 vLLM server с `max-model-len=8192`, затем
strict и support на одном 20-QA manifest и одном job-local cache. После live
strict extraction оба scoring run (`strict` и `support`) запускаются с
`--kg-cache-only`; cache miss или
изменение SHA graph cache немедленно завершает Job, поэтому KG не извлекаются
повторно. После live
run server останавливается. Runner повторяет strict и support с `--cache-only`
и требует одновременно:

- ноль live LLM calls;
- отсутствие cache misses;
- byte-identical strict/support `metrics.csv`;
- неизменные SHA-256 всех cache files.

Любое нарушение завершает Job ошибкой. Не запускайте две GPU Jobs одновременно
и не разделяйте strict/support на разные Jobs. Submitter сам получает JSON
список Jobs и fail-closed запрещает второй non-terminal `hallu-*` GPU Job.

## 6. Наблюдение и скачивание

Сохраните выданный `JOB_ID`:

```bash
.venv-datasphere/bin/datasphere --profile default project job get \
  --id <JOB_ID> --format json
```

- `PREPARING`: DataSphere создаёт контейнер и монтирует storage. Это уже не
  сборка Python environment внутри GPU Job.
- `EXECUTING`: смотрите Launch history, `pilot.stdout.log`, `pilot.stderr.log`,
  `vllm.log` и `gpu.csv`.
- `SUCCESS`, `ERROR`, `CANCELLED`: terminal status; GPU VM освобождена.

После terminal status:

```bash
mkdir -p "outputs/datasphere-results/$RUN_ID"
.venv-datasphere/bin/datasphere --profile default project job download-files \
  --id <JOB_ID> --with-logs --with-diagnostics \
  --output-dir "outputs/datasphere-results/$RUN_ID"
```

QA archive должен содержать как минимум `runtime-manifest.json`, оба freeze,
`shared-assets-preflight.json`, `gpu-runtime.json`, все четыре probe reports,
`qa_pilot_manifest.json`, `strict/`, `support/`, `cache-replay/`,
`cache-before.sha256`, `cache-after.sha256`, `comparison.json`,
`run_metadata.json`, `vllm.log`, `gpu.csv` и pilot logs.

## Быстрая диагностика

| Симптом | Проверка и действие |
|---|---|
| Runtime image не найден | Дождитесь remote workflow для exact commit; mutable tag использовать нельзя. |
| `--docker-image-id` отклонён локально | Для optional Project resource нужен ID строго `b[a-z0-9]{19}`. |
| Preflight: runtime built from another commit | Dockerfile был отрендерен не из текущего полного SHA. Соберите новый resource и повторите только CPU preflight. |
| Preflight: dependency/schema/S-BERT error | Не запускайте GPU. Исправьте Docker requirements/build, запушьте commit, соберите новый resource, повторите preflight. |
| vLLM не проходит healthcheck | Смотрите `gpu-runtime.json` и `vllm.log`; не переходите к probe/pilot. |
| Relation probe вернул bare triple | XGrammar/native schema contract не пройден. Не оборачивайте результат parser repair и не запускайте три QA. |
| KGGen или verifier probe не прошёл | Не отключайте LLM-clustering/typed output. Исправьте runtime и повторите один точный probe. |
| `failed_extractions.jsonl` не пуст | Gate не пройден; 20-QA pilot запрещён. |
| Cache-only replay пытается вызвать endpoint или меняет cache | Archive неполон или fingerprint не совпал. Job должен завершиться ошибкой; live повторный запуск не заменяет этот тест. |
| GPU 0% во время extraction 180/600 секунд | Watchdog завершит bounded/full extraction и сохранит diagnostics. Не перезапускайте вслепую. |

### Legacy, не использовать как текущую инструкцию

Старый runtime `env.python.type: manual` с vLLM `0.6.3.post1`, Outlines
`0.0.46`, LM Format Enforcer `0.10.6`, `pyairports` shim и runtime patch
зафиксирован только как история исходного сбоя. Он нарушал корневую relation
schema на реальном prompt и больше не является допустимым целевым runtime.
