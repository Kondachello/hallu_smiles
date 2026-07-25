# Ветка `zhenya` — быстрый инфраструктурный smoke-test (5 записей)

Эта ветка создана поверх `codex/experiment-framework-spec` как личный
быстрый прогон: проверить сквозной пайплайн (branch → CI-образ →
DataSphere Job → historical cache reuse → paired HalluGraph/GraphEval
predictions) на малом числе записей, прежде чем разбираться с частотой/
масштабом полноценного прогона. Это **не** research-эксперимент и не
заменяет `docs/vertex-100qa-hypothesis-report.md` или
`../../global_docs_SMILES/tasks/DASH_BOARD_HALLUGRAPH_VS_GRAPHEVAL.md`.

## Изменения относительно `codex/experiment-framework-spec`

1. **`--replay-count` по умолчанию `1 → 5`** в
   `scripts/render_datasphere_historical_qa_cache_replay_job.py` и
   `scripts/submit_datasphere_historical_qa_cache_replay.sh`. Значение
   по-прежнему можно переопределить явным флагом (допустимый диапазон
   1–100, как и раньше). См. также
   [`docs/datasphere-historical-qa-cache-replay.md`](datasphere-historical-qa-cache-replay.md#быстрый-минимальный-прогон-5-записей-ветка-zhenya).

2. **Ветка `zhenya` добавлена в триггеры CI**
   (`.github/workflows/datasphere-vertex-cpu-runtime-image.yml`), иначе push
   сюда не собирал бы Docker-образ и submit-скрипт бесконечно ждал бы
   несуществующий image digest.

3. **Фикс Windows bash-резолюции** в `scripts/validate_datasphere_job.py`
   (`_resolve_bash()`): голое имя `"bash"`, переданное в
   `subprocess.run(...)`, на Windows может резолвиться в
   `C:\Windows\System32\bash.exe` (легаси-лаунчер WSL) независимо от порядка
   `PATH` и рабочей директории, и падает сразу же, если WSL не настроен
   (`WSL 2 relay ... proxy server localhost` в stderr). Это чисто локальная
   проблема разработческой Windows-машины на этапе client-side валидации
   рендеренного YAML — сама Job внутри Docker-контейнера линуксовая, там
   этого класса ошибок нет. Резолвер сначала пробует
   `HALLU_BASH_EXE` (явный override), затем два стандартных пути Git for
   Windows, и только потом падает обратно на голое `"bash"` (для
   не-Windows окружений).

4. **Retry с backoff вокруг gateway-manifest fetch** в
   `scripts/run_datasphere_historical_qa_cache_replay.sh`. Раньше
   единственный `curl` к `/v1/hallu/manifest` не имел повторов вовсе — живой
   прогон (Job `bt1mpi501qj240rbjvsi`) упал на транзиентном `504` от Cloud
   Run gateway. Добавлена та же политика, что и для остальных live-вызовов
   в проекте: 429/5xx/сетевые ошибки повторяются с экспоненциальным backoff
   (5s→60s, джиттер 0–5s, до 5 попыток), 4xx падает сразу без повторов.

5. **Убран жёсткий "100-QA" хардкод, добавлена поддержка чекпоинтов любого
   размера (в т.ч. 750-QA)**. Раньше `HISTORICAL_CHECKPOINT_BASE` был
   буквальной строкой `.../qa-100-test-20-cv-5` в шаблоне Job, а резолвер
   (`resolve_datasphere_historical_qa_cache.py`) по умолчанию ожидал ровно
   100/80/20/5. Теперь:
   - Job получает `CHECKPOINT_PARENT` (родительский каталог
     `checkpoints/vertex-qa`, без конкретного размера) и `QA_SAMPLE_SIZE`
     (новый параметр `--qa-sample-size`, по умолчанию `100` для обратной
     совместимости).
   - `run_datasphere_historical_qa_cache_replay.sh` сначала аутентифицирует
     gateway-манифест (нужен его SHA для поиска), затем **широко** ищет все
     подходящие исторические чекпоинты под `CHECKPOINT_PARENT` (2 уровня
     вложенности, суффикс имени папки = SHA манифеста — тот же паттерн, что
     уже использует `run_datasphere_vertex_cpu_qa_pilot.sh`), читает
     `qa_sample.total/train/test/alpha_cv_folds` из `checkpoint-identity.json`
     **каждого** кандидата (не предполагает заранее), и фильтрует по
     `QA_SAMPLE_SIZE` — так можно явно выбрать между 100-QA и 750-QA
     чекпоинтами, даже если оба совпадают по gateway-манифесту.
   - Перенесён (без изменений) `scripts/resolve_datasphere_historical_cache_lineage.py`
     из общего паттерна, уже использующегося в `run_datasphere_vertex_cpu_qa_pilot.sh`
     этой же ветки (автор — Артемий Маслов, коммит `d74fed5`) — **не** портировано
     из ветки `new-metrics`, та ветка вообще не содержит парного
     HalluGraph/GraphEval сравнения и не трогалась.
   - `--replay-count` больше не ограничен диапазоном 1–100 — теперь
     `1 ≤ replay_count ≤ qa_sample_size`.
   - Каждый прогон discovery пишет `reports/historical_cache_discovery.json`
     (все проверенные кандидаты, почему каждый принят/отклонён) — это же
     артефакт для отладки (см. пункт 6).
   - Реестр `datasphere/historical_kg_cache_lineages.json` **не менялся**:
     единственная существующая запись матчится по `source_commit` +
     `gateway_manifest_sha256` + `client_runtime`, а не по числу записей, и
     по наблюдению остаётся валидной для чекпоинтов любого размера, если они
     были извлечены тем же коммитом/gateway.

6. **Локальный HTML-вьюер прогресса** (не часть детекторного пайплайна,
   чистая презентационная надстройка):
   - `scripts/render_historical_replay_progress_html.py` — читает
     `reports/progress.jsonl` + `predictions/raw_predictions.jsonl` +
     `reports/historical_cache_discovery.json` из уже скачанного (в т.ч.
     **частично**, для ещё выполняющейся Job) архива и рисует одну
     статическую HTML-страницу: таблицу response_id × {HalluGraph,
     GraphEval} со скорами и статусами, прогресс-бар, список кандидатов
     чекпоинта.
   - `scripts/watch_datasphere_historical_replay.sh <JOB_ID> <RUN_ID> [interval]` —
     периодически скачивает файлы Job (`download-files`) и перегенерирует
     HTML, пока статус не станет terminal (`SUCCESS`/`ERROR`/`CANCELLED`).
     `download-files` на ещё выполняющейся Job может честно вернуть "файлов
     пока нет" — это не ошибка цикла, просто следующая попытка.

## Как воспроизвести

См. раздел "Как отправить джобу заново" в `AGENT_HANDOFF.md` в корне рабочей
папки (на уровень выше `hallu_smiles/`), там же — установленные инструменты,
известные обходные пути для Windows и текущий статус последнего прогона.
