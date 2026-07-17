# DataSphere: памятка команде — запуск с первой попытки

> **Контекст работы.** Базовые прогоны я выполнял в ветке `qa-sample-test`.
> Новая проверяемая гипотеза находится в `new-metrics`: она сопоставляет
> исходную строгую графовую метрику с поддержкой отношения текстовым evidence
> и verifier. Любую новую экспериментальную ветку создаём от актуального
> `new-metrics`.

Эта инструкция — практический путь от локальной ветки до сохранённого отчёта.
Она написана после реальных ошибок первого запуска; не заменяйте шаги
«короткими» ручными командами. Полная техническая справка находится в
[datasphere-team-runbook.md](datasphere-team-runbook.md), а этот файл можно
целиком переслать исполнителю.

## Что уже подготовлено и что нельзя трогать

- Общие **Llama 3.1 8B** и **RAGTruth** уже лежат в общем Project storage.
  Их один раз подготовили CPU-сессией; каждый GPU Job читает эти файлы
  read-only и **не скачивает модель повторно**.
- В `shared/models` и `shared/ragtruth` разрешено хранить только модель,
  датасет и их manifests. Не удаляйте и не редактируйте их из рабочей Job.
- `HF_TOKEN` нужен только для редкого одноразового staging модели. В GPU Job,
  конфиги, git, логи и скриншоты он не попадает.
- Закреплённый `vllm==0.6.3.post1` менять нельзя без отдельной проверки:
  именно эта версия совместима с CUDA driver V100 `g1.1`. Диапазон вида
  `vllm>=...` может молча поставить несовместимую версию и потратить GPU-время
  до первой полезной операции.
- Пара `vllm==0.6.3.post1` и `lm-format-enforcer==0.10.6` требует ровно
  `transformers==4.45.2`. Это проверенная связка: плавающий диапазон поставил
  4.57, где удален `LogitsWarper`; сервер стартовал, но первый completion
  завершался HTTP 500. Transformers 5.x дополнительно несовместим с PyTorch
  2.4 ещё до старта vLLM. `outlines==0.0.46` закреплён отдельно: это
  constrained-decoding backend для полного вложенного JSON Schema KGGen.
- Локальный vLLM намеренно ограничен 8 192 токенами на V100. Поэтому KGGen
  должен получить `llm.max_tokens: 256`: его дефолт 16 000 переполняет окно
  ещё до генерации. Скрипт делает дешёвый completion smoke-check и затем
  KGGen/DSPy typed-output **и clustering** probe с timeout 180 секунд,
  прежде чем запускать все 20 QA. Кластеризация — часть KGGen, не fallback.
- Для vLLM 0.6.3 выбран `--guided-decoding-backend outlines`: он ограничивает
  **тот же полный Pydantic JSON Schema** KGGen, включая вложенный
  `relations: list[Relation]`. У `lm-format-enforcer==0.10.6` после успешного
  простого probe была подтверждена ошибка на реальном nested schema: он мог
  вернуть один bare `{"subject", "predicate", "object"}` вместо корневого
  `{"relations": [...]}`. Это не исправляют постобработкой и не обходят
  отключением typed extraction или LLM clustering.

  `outlines==0.0.46` требует `pyairports==0.0.1`, а этот PyPI distribution не
  содержит импортируемого Python-модуля. Поэтому Job добавляет в `PYTHONPATH`
  проверенный репозиторный shim `datasphere/runtime_shims/pyairports` с пустым
  неиспользуемым списком аэропортов. CPU preflight импортирует именно этот shim
  и `outlines.integrations.vllm.JSONLogitsProcessor` до выделения V100.
  `patch_datasphere_lmfe_bool_schema.py` остаётся обязательным audit-патчем
  точного pinned dependency vLLM: он применяет upstream five-line fix только в
  ephemeral Job venv и проверяет closed schema. Он не меняет schema, KGGen или
  кластеризацию.

## Короткая схема

```text
qa-sample-test ── baseline-прогоны
        │
        └── new-metrics ── строгий baseline + новая support-гипотеза
                 │
                 └── ваша ветка → push commit → CPU preflight
                                                     │
                                      cluster-probe (3 QA, при смене runtime)
                                                     │
                                                     └── GPU QA Job → archive → отчёт
```

После новой зависимости или зависания сначала запустите `cluster-probe-g1`:
он поднимает ту же Llama, но извлекает и **кластеризует** только первые три
записи фиксированного manifest. Это дешёвая проверка границы, на которой
раньше возникал stall. Успешный probe не является метрикой и не заменяет
полный пилот. В логе probe должны быть пары `cluster:start` / `cluster:done`;
если один вызов зависнет, появится Python thread dump, а Job остановится через
180 секунд нулевого GPU вместо десятиминутного простоя полного pilot.

Один полный GPU Job запускает этапы в одном процессе: сначала strict baseline,
затем support-вариант. Они используют **одинаковые** 20 `(C,Q,A)`, один
selection manifest, KG cache и запущенный vLLM. Поэтому не создавайте две
GPU Job «для baseline и new metrics» — это будет дороже и исказит сравнение.

## 0. Локальная подготовка (делается один раз на ноутбук)

Из корня репозитория:

```bash
python3.12 -m venv .venv-datasphere
.venv-datasphere/bin/python -m pip install --upgrade pip datasphere
# Установить YC CLI по официальной инструкции Yandex Cloud.
# Если он уже лежит в репозитории, достаточно:
export PATH="$PWD/.tools/yc:$PATH"
yc init
```

Для выданного/federated аккаунта запускайте **обычный** `yc init`, без
`--username`. Если после входа CLI пишет, что доступных cloud нет, **не
создавайте cloud и новый DataSphere Project**. Это ожидаемо для этого типа
доступа: проверка прав делается так (ID берём у координатора проекта):

```bash
.venv-datasphere/bin/datasphere --profile default project get --id <PROJECT_ID>
```

Если команда возвращает проект, доступ настроен. Ни токены, ни секреты,
ни вывод `yc` не отправляем в чат, Git или screenshots.

## 1. Подготовить свою ветку

```bash
git fetch origin
git switch new-metrics
git pull --ff-only origin new-metrics
git switch -c <your-branch>
# правки + тесты
git add <только-нужные-файлы>
git commit -m "..."
git push -u origin <your-branch>
git rev-parse HEAD
```

DataSphere клонирует только commit, уже опубликованный в GitHub. Не начинайте
от `main`, старого `qa-sample-test` или локального незапушенного состояния.
Для каждого запуска выбирайте уникальный lowercase `RUN_ID` без `/`, например
`my-hypothesis-a1b2c3d-20260717`.

## 2. Обязательный бесплатный CPU preflight

Это проверяет доступ к общей модели и данным **до** аренды GPU. Он должен
первым находить неправильную ветку, отсутствующий ready-marker, неверный
model SHA и данные. Запускать GPU без успешного preflight нельзя.

```bash
PROJECT_ID=<PROJECT_ID>
BRANCH=<your-branch>
RUN_ID=preflight-<your-branch>-<date>

PYTHON_BIN=.venv-datasphere/bin/python \
DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
bash scripts/submit_datasphere_job.sh \
  --kind preflight --project-id "$PROJECT_ID" --branch "$BRANCH" --run-id "$RUN_ID"
```

Сохраните выданный `JOB_ID`, дождитесь `SUCCESS` и проверьте JSON preflight:
`status: ready`, HF revision и ненулевой `model_bytes_checked`.

## 3. Единственная команда GPU-прогона

Только после `SUCCESS` preflight:

```bash
RUN_ID=<your-branch>-<short-sha>-<date>

PYTHON_BIN=.venv-datasphere/bin/python \
DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
bash scripts/submit_datasphere_job.sh \
  --kind qa-pilot-g1 --project-id "$PROJECT_ID" --branch "$BRANCH" --run-id "$RUN_ID"
```

Не редактируйте `datasphere/jobs/*.template.yaml` ради одного запуска и не
пишите руками `datasphere project job execute`. Helper сам:

1. проверяет, что commit опубликован в выбранной GitHub-ветке;
2. рендерит YAML;
3. валидирует shell/YAML/requirements локально;
4. создаёт Job только после этих проверок.

Если helper завершился ошибкой, Job не создан и GPU units не потрачены.
Исправляем проблему в ветке, коммитим, пушим и повторяем ту же команду.

GPU Job использует `g1.1` (V100 32 GB), FP16, `max-model-len=8192`, один
vLLM server, и жёсткий лимит 3 часа. `trap` останавливает vLLM при успехе,
ошибке, отмене или timeout. После terminal status DataSphere освобождает GPU
автоматически.

## 4. Наблюдение: каждые 10 минут, без платного простоя

Сразу после submit запишите `JOB_ID` в task/issue. Проверять нужно каждые
10 минут (агенту можно дать инструкцию ниже):

```bash
.venv-datasphere/bin/datasphere --profile default project job get \
  --id <JOB_ID> --format json
```

Значения статуса означают следующее:

| Статус | Действие |
|---|---|
| `PREPARING` | Среда подготавливается; это ещё не полезная GPU-работа. Не отменяйте только из-за очереди, но не считайте это вычислением. |
| `EXECUTING` | Job активна. После старта проверить `gpu-runtime.json`, healthcheck vLLM, а в конце — `gpu.csv` и `vllm.log`. |
| `SUCCESS` | GPU уже освобождён. Скачать результаты, прочитать отчёт, остановить мониторинг. |
| `ERROR` / `CANCELLED` | GPU уже освобождён. Скачать diagnostics, зафиксировать точную ошибку; не перезапускать вслепую. |

В web-интерфейсе: **DataSphere Jobs → Launch history**. На macOS
`project job attach` иногда засыпает терминал локальными gRPC warnings; это не
диагностика Job. В таком случае используйте `project job get` и Launch history.
`download-files` у активной Job может ответить, что файлов ещё нет — это не
ошибка и не повод перезапускать Job.

Если Job долго `EXECUTING`, но vLLM не проходит healthcheck или нет GPU
activity после его старта, отменяйте **только конкретный** Job:

```bash
.venv-datasphere/bin/datasphere --profile default project job cancel \
  --id <JOB_ID> --graceful
```

Не используйте широкую кнопку «Отменить задания», если нужно остановить лишь
один запуск. Перед отменой сохраните точный симптом и логи; новый запуск
делается только после устранения причины.

## Готовая инструкция для агента/исполнителя

Скопируйте этот блок в задачу агенту, который наблюдает запуск:

```text
Работай только в ветке new-metrics или её опубликованной дочерней ветке.
Не создавай DataSphere Project/cloud, не меняй shared/models и shared/ragtruth,
не передавай HF_TOKEN в Job и не меняй pins vllm==0.6.3.post1 и
transformers==4.45.2 (вместе с lm-format-enforcer==0.10.6 и
outlines==0.0.46). Не удаляй `datasphere/runtime_shims/pyairports`: он нужен
Outlines для полного nested JSON Schema KGGen и не является обходом extraction
или clustering.

До GPU Job требуй успешный CPU preflight: в archive должны быть `preflight.json`,
`runtime-dependencies.json` и `outlines-backend.json` со status `ready`.
Последние два файла создаются из тех же точных requirements, что и GPU Job, и
импортируют vLLM, Transformers, lm-format-enforcer и Outlines c репозиторным
shim без модели и без GPU. Для запуска используй только
scripts/submit_datasphere_job.sh с уникальным RUN_ID; не редактируй Job YAML
и не запускай datasphere project job execute вручную.

После submit сохраняй JOB_ID и проверяй статус каждые 10 минут командой
datasphere project job get --id <JOB_ID> --format json. PREPARING означает
ожидание, EXECUTING — проверяй, что vLLM проходит healthcheck и GPU не простаивает.
При SUCCESS/ERROR/CANCELLED сразу скачай outputs, logs и diagnostics, прочитай
gpu-runtime.json, run_metadata.json, comparison.json, gpu.csv и vllm.log,
сделай краткий отчёт и останови мониторинг. Не запускай повторную Job без
разбора точной причины terminal failure. Не держи Jupyter или GPU включёнными
ради чтения результатов.
```

## 5. Скачать и проверить результаты после terminal status

```bash
mkdir -p "outputs/datasphere-results/<RUN_ID>"
.venv-datasphere/bin/datasphere --profile default project job download-files \
  --id <JOB_ID> --with-logs --with-diagnostics \
  --output-dir "outputs/datasphere-results/<RUN_ID>"
```

Обязательные файлы QA-архива:

- `shared-assets-preflight.json` — общие модель/датасет прочитаны;
- `gpu-runtime.json` — CUDA smoke-check до vLLM;
- `vllm.log` и `gpu.csv` — запуск LLM и загрузка GPU;
- `run_metadata.json`, `qa_pilot_manifest.json` — commit, конфигурация и
  детерминированные 20 QA;
- `strict/` и `support/` — независимые результаты двух режимов;
- `comparison.json` и `comparison.md` — train OOF AUC, test AUC и сравнение;
- audits, KG/verdict cache statistics, wall-clock и diagnostics.

На 4 test примерах AUC дискретный и нестабильный. В отчёте всегда интерпретируем
его вместе с 5-fold train OOF AUC, а не как окончательную оценку метода.

## 6. JupyterLab: когда нужен и когда его освобождать

Dedicated JupyterLab `c1.4` нужен только для одноразового staging общей модели
и RAGTruth. Внутри Jupyter вводится **Python-код**, не shell-фрагмент. После
успешных строк `[ok] shared model ready` и `[ok] shared RAGTruth ready` его
нужно остановить: UI «Остановить JupyterLab и ВМ» освобождает CPU/RAM и не
ломает уже выполняющуюся DataSphere Job. Он также не удаляет Project storage,
модель, RAGTruth или архивы Job.

Важная оговорка: UI предупреждает, что конфигурации Jupyter будут очищены.
Если кому-то ещё нужна сама Jupyter-сессия, сначала согласуйте остановку. Если
она больше не нужна — не держите её «на всякий случай».

Падение Python kernel в Jupyter после staging не влияет на уже записанные
shared assets и не влияет на отдельные Jobs.

## 7. Карта уже встреченных ошибок

| Симптом | Причина | Что делать правильно |
|---|---|---|
| `SyntaxError` в Jupyter | В Python-ячейку вставили shell/CLI-текст. | Вставлять только Python с `subprocess.run(...)` из полной runbook. |
| `KeyError: 'DS_PROJECT_HOME'` в Jupyter | `DS_PROJECT_HOME` есть только внутри Job. | В Jupyter использовать `pathlib.Path('hallu_smiles') / 'shared'`; в Job — `$DS_PROJECT_HOME/...`. |
| `Python crashed` в Jupyter | Упал kernel/сессия, а не GPU Job. | Проверить наличие ready-marker; не считать уже завершённый staging потерянным. |
| `yc init` не принимает выданный аккаунт / «no clouds available» | Federated DataSphere account не обязан иметь обычный cloud. | `yc init` без `--username`; затем проверить существующий DataSphere Project, ничего нового не создавать. |
| `file set was not found` при создании Job | CLI принял shell `set ...` за имя файла. | Job command начинается с `bash -lc`, это уже зашито в template. |
| `cmd contains variable not presented in config` | `${PWD}` / `${RUN_ROOT}` ошибочно распознаны как YAML-подстановки. | В shell использовать `$PWD`, `$RUN_ROOT`; не править template вручную. |
| CPU preflight завершается сразу с `PYTHONPATH: unbound variable` | В чистом manual Job включён `set -u`, но `PYTHONPATH` может отсутствовать. | Не экспортировать `...:$PYTHONPATH` без guard. Template проверяет `printenv PYTHONPATH` и задаёт shim directory отдельно, если переменная отсутствует. |
| `Python root modules not found` | В manual Job не указали локальные Python paths/requirements. | Использовать helper; он добавляет `local-paths` и requirements. |
| Ошибка requirements parser | В requirements были комментарии или `-r`. | Оставлять только прямые PEP 508 зависимости в `requirements.datasphere.txt`. |
| Аргумент `--shared-root` стал отдельной командой | Некорректный folded YAML/отступ. | Не редактировать сгенерированную shell-команду; helper её валидирует. |
| vLLM не стартует: driver/CUDA too old | Установилась новая несовместимая версия vLLM. | Не использовать `>=`; сохранить `vllm==0.6.3.post1` и смотреть `gpu-runtime.json`. |
| vLLM ждёт healthcheck, а в логе `cannot import name 'DTensor'` | Resolver поставил Transformers 5.x, несовместимый с PyTorch 2.4. | Сохранить прямой pin `transformers==4.45.2`; не убирать его при обновлении requirements. |
| `/v1/chat/completions` отвечает `400`, а `failed_extractions.jsonl` содержит `ContextWindowExceededError` | KGGen без явного лимита просит до 16 000 output tokens, но vLLM для V100 ограничен 8 192. | Сохранить `llm.max_tokens: 256` в runtime config и completion smoke-check до QA. |
| Completion smoke-check отвечает `500`, в `vllm.log` — `ModuleNotFoundError: pyairports` | Outlines 0.0.46 импортирует дефектный `pyairports==0.0.1`, где нет Python-модуля. | Не переключать backend на raw или LME. Проверить, что Job экспортирует `PYTHONPATH=.../datasphere/runtime_shims` и preflight создал `outlines-backend.json` со `status: ready`. |
| Completion smoke-check отвечает `500`, в `vllm.log` — `cannot import name 'LogitsWarper'` | Плавающий `transformers>=...` поставил версию 4.57, несовместимую с `lm-format-enforcer`. | Нужен точный, а не диапазонный pin: `transformers==4.45.2`; затем заново пройти CPU preflight. |
| Guided-JSON probe отвечает `500`, а в `vllm.log` — `AttributeError: 'bool' object has no attribute 'get'` из `lmformatenforcer/.../jsonschemaobject.py` | `lm-format-enforcer==0.10.6` не умеет boolean-значение `additionalProperties: false` в закрытой JSON Schema, которую vLLM передаёт для typed KGGen extraction. Это происходит **до** QA/KGGen и после загрузки модели. | Не менять vLLM dependency pin. `patch_datasphere_lmfe_bool_schema.py` делает только upstream five-line backport в ephemeral Job venv и сохраняет JSON-отчёт; CPU preflight затем создаёт настоящий `JsonSchemaParser(... additionalProperties: false)`. Не отключать clustering или constrained decoding. |
| Простой guided-JSON probe проходит, но реальный KGGen relations падает с `Expected output fields [relations], actual []`, а ответ — bare `{"subject", "predicate", "object"}` | `lm-format-enforcer==0.10.6` не сохраняет корневую структуру у nested `$defs` Pydantic Schema. Это не проблема модели, VRAM или LLM clustering. | Использовать выбранный `outlines` backend с тем же полным schema и pyairports shim; сначала пройти CPU preflight и `cluster-probe-g1`. Не добавлять эвристику, которая оборачивает bare triple. |
| `kggen-probe.json` не появляется или Job завершается до QA | DSPy 3.x/LiteLLM drift совместим по import, но не по KGGen structured-output path. | Сохранить locked `kg-gen==0.4.0`, `dspy==2.6.27`, `litellm==1.60.4`; probe должен пройти до extraction. |
| Несовместимость Python-зависимостей обнаружена до GPU | CPU preflight импортирует ровно тот же набор requirements и пишет `runtime-dependencies.json` и `outlines-backend.json`. | Не запускать GPU, пока `preflight.json`, `runtime-dependencies.json` и `outlines-backend.json` имеют `status: ready`; исправить pin, запушить commit и повторить только CPU preflight. |
| GPU остаётся 0% на KG extraction | Клиент KGGen/DSPy мог застыть после ответа vLLM при обработке cluster-процедуры. | Проверить `qa-reference-probe.json` и `vllm.log`; кластеризацию не отключать в официальном запуске. Сохранить pins `kg-gen==0.4.0`, `dspy==2.6.27`, `litellm==1.60.4`, `pydantic==2.10.6`, LLM timeout и serial KGGen. Сначала пройти `cluster-probe-g1` на трёх фиксированных QA. Watchdog завершит только extraction через 600 секунд сплошного нулевого GPU; после этого читать diagnostics, а не перезапускать вслепую. |
| В error Job нет `vllm.log` / archive | При аварийном выходе DataSphere не передал путь output archive в shell. | Template задаёт fallback `ARTIFACT_ARCHIVE` до trap. Не удалять его: частичный archive должен собираться и при error/cancel. |
| Логи `attach` шумные или пустые | Локальный macOS gRPC клиент нестабилен. | Смотреть `job get` и Launch history; архив скачивать после terminal. |
| Повторная загрузка модели / нехватка диска | GPU Job пытается staging/download. | GPU Job только читает ready-marker; staging — лишь одноразово на c1.4. |
| Непонятно, где результат | Job output — не Git и не shared model folder. | Скачивать архив в `outputs/datasphere-results/<RUN_ID>/`. |

## Нельзя делать никогда

- Не создавать новый Project/cloud и не подменять credentials команды личными.
- Не отправлять в Git веса, `HF_TOKEN`, API keys, `.tools/`,
  `.venv-datasphere/`, outputs, caches или diagnostics с секретами.
- Не смешивать `RUN_ID`, manifests, caches и результаты двух веток.
- Не запускать `huggingface-cli download` или runtime `pip install` в GPU Job.
- Не освобождать shared storage и не трогать `ready`/`active-model.json` без
  отдельного согласованного обновления модели.
- Не держать Jupyter или GPU Job запущенными только ради просмотра CSV/лога.

Если инструкция и фактический интерфейс расходятся, сначала остановитесь,
снимите точный текст ошибки и приложите `JOB_ID`, ветку, commit и `RUN_ID`.
Это почти всегда позволяет исправить запуск без траты ещё одного GPU-слота.
