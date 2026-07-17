# DataSphere: общий Llama 3.1 8B и воспроизводимые Jobs

Эта памятка — единственный путь запуска экспериментов команды в проекте
`Online_project19_1`. Она разделяет одноразовую подготовку общих assets и
изолированные Jobs для веток. Не создавайте новый cloud или новый Project:
всегда используйте выданный проект и его ID.

## Что общее, а что изолировано

```text
Dedicated JupyterLab c1.4 (одноразовая запись)             DataSphere Job (только чтение)
/home/jupyter/project/hallu_smiles/shared/        ->       $DS_PROJECT_HOME/hallu_smiles/shared/
  models/.../<HF SHA>/{manifest,ready,weights}              model weights
  ragtruth/{jsonl,manifest}                                  RAGTruth

ветка/commit/run-id Job -> /job/qa-pilot-artifacts/... -> archive Job output
                                    cache/kg, cache/verdicts, audit, logs, reports
```

Общий storage уже содержит Llama 3.1 8B и RAGTruth. В нём разрешены только
модель, данные и их manifests. Никто не кладёт туда output, manifest выборки,
KG cache или verdict cache и не удаляет файлы из `shared/models`/
`shared/ragtruth`.

Для Jobs shared storage монтируется read-only. Поэтому каждая ветка получает
свои кэши, audit и отчёты в своём Job output, но не скачивает Llama заново.

## Разовый staging модели и данных

Нужен только если в shared storage отсутствует `active-model.json` или
ready-marker, либо команда сознательно меняет HF revision. Для обычной ветки
его **не запускать**.

1. В DataSphere создайте/откройте `c1.4` JupyterLab. Только для этой сессии
   добавьте Project secret `HF_TOKEN`; он не передаётся ни в один GPU Job.
2. Вставляйте в ячейку Jupyter именно Python, не shell-команду. `m ml.; ...`
   или другие shell-фрагменты в Python-ячейке дадут `SyntaxError`.
3. `DS_PROJECT_HOME` — переменная **Job**, в Dedicated JupyterLab её нет. Не
   используйте `os.environ["DS_PROJECT_HOME"]`: это приводит к `KeyError`.
4. Запустите ровно эту ячейку; секрет не печатается:

   ```python
   import pathlib
   import subprocess
   import sys

   repo = pathlib.Path("hallu_smiles")
   subprocess.run(
       [
           sys.executable,
           "scripts/stage_datasphere_shared_assets.py",
           "--shared-root", str(repo / "shared"),
           "--model-id", "meta-llama/Meta-Llama-3.1-8B-Instruct",
       ],
       cwd=repo,
       check=True,
   )
   ```

5. Успех — строки `[ok] shared model ready`, `[ok] shared RAGTruth ready`.
   Скрипт создаёт ready-marker только после полной загрузки. При следующем
   вызове он выводит `[skip]`, а не скачивает веса повторно.
6. После успеха JupyterLab больше не требуется. Освобождайте `c1.4`, чтобы
   не платить за простаивающую ВМ. В UI действие «Остановить JupyterLab и ВМ»
   также очищает Jupyter-конфигурации; оно не удаляет shared Project storage
   и не отменяет уже запущенные DataSphere Jobs. Если настройки Jupyter нужны
   кому-то прямо сейчас, сначала согласуйте освобождение ВМ.

Упавший Jupyter Python kernel после staging не влияет на уже записанные assets
и не влияет на Jobs: Job выполняется как отдельная среда.

## Локальная настройка CLI

На рабочем Mac/Linux выполняйте из корня репозитория:

```bash
python3.12 -m venv .venv-datasphere
.venv-datasphere/bin/python -m pip install --upgrade pip datasphere
# yc установить официальным способом: https://yandex.cloud/en/docs/cli/operations/install-cli
yc init
```

Для выданного/federated DataSphere-аккаунта запускайте plain `yc init`, без
`--username`. Если `yc init` после входа сообщает, что доступных clouds нет,
это не повод создавать cloud и не блокирует Jobs. Проверка авторизации для
нашего случая — существующий DataSphere Project:

```bash
.venv-datasphere/bin/datasphere --profile default project get --id <PROJECT_ID>
```

Если `yc` установлен рядом с репозиторием, а не в system PATH, сначала:

```bash
export PATH="$PWD/.tools/yc:$PATH"
```

Не записывайте OAuth token, `HF_TOKEN`, API key или их значения в Git, YAML,
output или скриншоты.

## Единственная команда запуска ветки

Не редактируйте `datasphere/jobs/*.template.yaml` для конкретного запуска и не
вводите `datasphere project job execute` вручную. Используйте helper: он
проверяет, что commit уже находится в публичной ветке GitHub, рендерит YAML,
проверяет его локально и лишь затем создаёт Job.

`RUN_ID` должен быть уникальным, lowercase и без `/`; включайте в него ветку,
короткий commit и дату. Например `new-metrics-9e8c04e-20260717`.

```bash
PROJECT_ID=<PROJECT_ID>
BRANCH=new-metrics
RUN_ID=preflight-new-metrics-9e8c04e-20260717

PYTHON_BIN=.venv-datasphere/bin/python \
DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
bash scripts/submit_datasphere_job.sh \
  --kind preflight --project-id "$PROJECT_ID" --branch "$BRANCH" --run-id "$RUN_ID"
```

Сначала всегда дождитесь terminal `SUCCESS` CPU preflight. Его JSON должен
содержать `status: "ready"`, HF revision и ненулевой `model_bytes_checked`.
Только затем запускайте GPU:

```bash
RUN_ID=new-metrics-9e8c04e-20260717
PYTHON_BIN=.venv-datasphere/bin/python \
DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
bash scripts/submit_datasphere_job.sh \
  --kind qa-pilot-g1 --project-id "$PROJECT_ID" --branch "$BRANCH" --run-id "$RUN_ID"
```

GPU Job неизменно использует `g1.1` (V100 32 GB), FP16,
`max-model-len=8192`, один vLLM server для двух стадий и hard timeout в 3 часа
(777 600 units + максимум 60 секунд graceful shutdown). Внутри него порядок
фиксирован: strict baseline создаёт 20-QA manifest и KG cache; support-run
использует их повторно. При normal exit, error, cancel или timeout `trap`
останавливает vLLM. После terminal status DataSphere освобождает GPU VM сам;
держать Job для просмотра отчёта не нужно.

Перед импортом KGGen Job выставляет `LITELLM_LOCAL_MODEL_COST_MAP=true`: локальному
vLLM не нужна скачиваемая LiteLLM-карта цен, а внешняя сеть не должна быть частью
критического пути. KG extraction запускается отдельным шагом с liveness watchdog:
если `gpu.csv` показывает 600 секунд подряд нулевой utilisation, завершается только
этот шаг, а exit-trap всё равно упаковывает частичные `vllm.log`, `gpu.csv`, manifest
и diagnostics. CPU-heavy scoring не контролируется этим watchdog, поэтому у него нет
ложных отмен после успешно законченного extraction.

`requirements.datasphere.txt` намеренно закрепляет `vllm==0.6.3.post1`: эта
версия использует PyTorch 2.4/CUDA 12.1, совместимые с CUDA 12.2 driver `g1.1`.
Не заменяйте pin на диапазон версий: новый vLLM может потребовать более свежий
driver. Также сохраняйте точный `transformers==4.45.2`: эта версия проверена
вместе с `lm-format-enforcer==0.10.6` и `outlines==0.0.46`. Плавающий диапазон поставил 4.57, где
удален `LogitsWarper`, и первый completion завершался HTTP 500. Transformers
5.x дополнительно требует API более нового PyTorch и ломает импорт vLLM 0.6.3
ещё до healthcheck. До загрузки модели Job записывает `gpu-runtime.json` с CUDA smoke-check,
версией PyTorch/CUDA, GPU и compute capability.

Нельзя оставлять плавающими и KGGen-клиентские зависимости: для текущей
ветки зафиксированы `kg-gen==0.4.0`, `dspy==2.6.27` и
`litellm==1.60.4`, `pydantic==2.10.6`. DSPy 3.x проходит простой import-check, но меняет
typed-output path; на практике после нескольких запросов KGGen переставал
подавать новые запросы в vLLM, а V100 простаивал. CPU preflight проверяет
всю эту матрицу именно как lock, а не только как набор импортируемых пакетов.

`max-model-len=8192` — сознательный лимит, позволяющий Llama 8B работать на
V100 32 GB. Для KGGen в Job обязательно выставляется `llm.max_tokens=256`:
без него KGGen 0.4 допускает до 16k output tokens, и vLLM отклоняет запросы с
`ContextWindowExceededError`. Сразу после `/health` Job выполняет один
двухтокенный `/v1/chat/completions` smoke-check, а затем отдельный маленький
**KGGen/DSPy probe** с typed-output и LLM-clustering. Второй probe проверяет
именно тот путь, который использует extractor; если он зависнет или
завершится ошибкой, QA-pilot не начнётся. Внешний `timeout` probe ограничен
180 секундами, поэтому несовместимость не превратится в три часа оплачиваемого
простоя.

Для vLLM 0.6.3 Job задаёт `--guided-decoding-backend outlines`: он выполняет
constrained decoding по **тому же полному Pydantic JSON Schema** KGGen,
включая вложенный `relations: list[Relation]`. Это принципиально: у
`lm-format-enforcer==0.10.6` после успешного простого JSON probe реальная
relation extraction вернула bare `{"subject", "predicate", "object"}` вместо
корневого `{"relations": [...]}`. Эту ошибку не исправляют постобработкой,
не отключают typed extraction и не отключают KGGen LLM clustering.

Outlines 0.0.46 по ошибке зависит от `pyairports==0.0.1`, а этот distribution
не содержит импортируемого Python-модуля. Job до старта vLLM добавляет
`datasphere/runtime_shims` в `PYTHONPATH`; shim поставляет только пустой
неиспользуемый список аэропортов, чтобы импорт Outlines работал. CPU preflight
обязан записать `outlines-backend.json` со `status: ready`, импортировав
`outlines.integrations.vllm.JSONLogitsProcessor` через тот же shim. Не удаляйте
shim и не меняйте backend без целевого smoke-check.

`lm-format-enforcer==0.10.6` остаётся pinned dependency vLLM 0.6.3. Перед
стартом server `patch_datasphere_lmfe_bool_schema.py` применяет только upstream
five-line fix для parsing `additionalProperties: false` в ephemeral Job venv и
пишет audit report. Это сохраняет совместимость pinned runtime, но не
подменяет выбранный Outlines backend.

Поскольку `lm-format-enforcer==0.10.6` импортирует `LogitsWarper`, нельзя
оставлять диапазон версий Transformers: на практике resolver выбрал 4.57, где
этот символ удален. Обязательная проверенная тройка —
`vllm==0.6.3.post1`, `lm-format-enforcer==0.10.6`,
`outlines==0.0.46`, `transformers==4.45.2`.

CPU preflight теперь не только проверяет ready-marker, SHA и размеры shared
assets. Он устанавливает **тот же exact requirements набор**, что GPU Job,
импортирует `vllm`, `transformers`, `lmformatenforcer`, его Transformers
adapter, `litellm`, `dspy`, `kg_gen` и Outlines с репозиторным pyairports shim,
а затем сохраняет `runtime-dependencies.json` и `outlines-backend.json`. Это не загружает 8B
веса и не резервирует GPU, но ловит resolver/import-конфликты до дорогого
запуска. GPU Job разрешён только когда `preflight.json`,
`runtime-dependencies.json` и `outlines-backend.json` имеют `status: ready`.

Для локального vLLM Job дополнительно принудительно задаёт
`llm.concurrency: 1` и `extraction.serial_chunking: true`. KGGen 0.4 создаёт
свой внутренний pool при `chunk_size`; совместно с внешним response pool он
разделяет один DSPy client и может зависнуть после части запросов. Serial режим
сохраняет split и aggregate, но устраняет nested concurrency. В профиле
DataSphere сохраняет KGGen LLM-clustering включённой: это часть метода,
канонизирующая entity/predicate aliases, и raw-only граф не считается
воспроизводимым результатом. После двух успешных reference-графов был
воспроизведён Pydantic stall в client-side обработке одного cluster response.
Поэтому перед полным 20-QA запуском используется отдельный `cluster-probe-g1`:
тот же vLLM, те же pins и clustering, но лишь первые три записи fixed manifest
и лимит 1 час. В этом probe extractor вызывает официальный `KGGen.cluster()`
явно после raw extraction — алгоритм не подменяется, но логи получают
`cluster:start`/`cluster:done`. Через 120 секунд одного зависшего вызова
печатается безопасный Python thread dump, а watchdog останавливает probe после
180 секунд нулевого GPU; в полном pilot это остаётся 600 секунд. Только успех
probe разрешает полный pilot. Если `vllm.log`/
`gpu.csv` показывают несколько минут 0% utilisation после завершённых
запросов, отменяйте конкретную Job: такой простой не является полезным
вычислением.

## Что именно pre-submit проверяет

`scripts/validate_datasphere_job.py` запускается helper-ом автоматически. Он
ловит до создания Job ошибки, с которыми мы столкнулись на первом запуске:

- shell command обязан начинаться с `bash -lc`, а не с `set -Eeuo pipefail`;
  иначе CLI ошибается `file set was not found`;
- в Job YAML нельзя писать `${PWD}`/`${RUN_ROOT}`: CLI считает их собственными
  переменными и выдаёт `cmd contains variable not presented in config`;
- `env.python.local-paths` и `requirements-file` обязательны для manual shell
  Job, иначе CLI не может найти Python root module;
- `requirements.datasphere*.txt` допускают только прямые PEP 508 requirements:
  без комментариев и без `-r requirements.txt`;
- аргументы Python-команд в folded YAML не должны быть вынесены на дополнительно
  отступленные строки, иначе `--shared-root` становится отдельной командой;
- template не может содержать `huggingface-cli download` и `pip install`.

Если helper завершился ошибкой, Job **не создан** и units не потрачены. Исправьте
код/ветку, выполните `git commit && git push`, затем повторите одну команду.

## Наблюдение, остановка и артефакты

После submit сохраните выданный `JOB_ID`.

```bash
.venv-datasphere/bin/datasphere --profile default project job get --id <JOB_ID> --format json
```

- `PREPARING` — подготовка среды; GPU ещё не следует считать полезно занятым.
- `EXECUTING` — Job запущен. В браузере смотрите **DataSphere Jobs → Launch
  history**; после vLLM start проверяйте `vllm.log`/`gpu.csv` в результате.
  Через CLI live snapshot попадает в временную папку, напечатанную как
  `logs file path`; там `gpu_stats.tsv` позволяет подтвердить непрерывные 0%.
- `SUCCESS`, `ERROR`, `CANCELLED` — terminal. GPU VM автоматически освобождена.

На macOS `project job attach` иногда показывает локальные gRPC warnings вместо
полезного tail. В этом случае `project job get` и Launch history надёжнее.
`download-files` во время выполнения преднамеренно отвечает, что забирать ещё
нечего; это не ошибка Job. После terminal status:

```bash
mkdir -p "outputs/datasphere-results/<RUN_ID>"
.venv-datasphere/bin/datasphere --profile default project job download-files \
  --id <JOB_ID> --with-logs --with-diagnostics \
  --output-dir "outputs/datasphere-results/<RUN_ID>"
```

В архиве QA Job обязательны `shared-assets-preflight.json`, `gpu-runtime.json`,
`vllm.log`, `gpu.csv`, `run_metadata.json`, `qa_pilot_manifest.json`, `strict/`,
`support/`, `comparison.json` и `comparison.md`. Если Job уже `EXECUTING`, но
из log видно, что vLLM не прошёл healthcheck или `gpu_stats.tsv` показывает
более 10 минут 0% во время extraction,
отмените именно этот Job: `datasphere project job cancel --id <JOB_ID> --graceful`.
Не нажимайте широкую кнопку «Отменить задания», если хотите остановить только
JupyterLab.

## Правила параллельной работы

- Один Job = один опубликованный commit = один `RUN_ID` = один output archive.
- Любая ветка сначала делает CPU preflight, потом GPU Job.
- Обычный Job берёт модель только через `active-model.json` и ready-marker;
  `HF_TOKEN` в Job отсутствует.
- Не меняйте общие assets без согласования. Новая revision становится active
  только после полного staging и нового CPU preflight.
- После завершения скачивайте archive и освобождайте вычисления; результаты
  хранятся отдельно по `outputs/<branch>/<commit>/<run-id>/` локально или в
  назначенном командном persistent storage.
