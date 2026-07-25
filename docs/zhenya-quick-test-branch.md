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

## Как воспроизвести

См. раздел "Как отправить джобу заново" в `AGENT_HANDOFF.md` в корне рабочей
папки (на уровень выше `hallu_smiles/`), там же — установленные инструменты,
известные обходные пути для Windows и текущий статус последнего прогона.
