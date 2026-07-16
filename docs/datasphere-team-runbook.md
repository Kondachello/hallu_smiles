# DataSphere: общая Qwen и безопасные запуски веток

Эта памятка описывает единственный общий экземпляр Qwen и изолированные Jobs
для экспериментов. Она рассчитана на проект `Online_project19_1` в одной
DataSphere community.

## Общие неизменяемые assets

В Project storage используются только эти пути:

```text
$DS_PROJECT_HOME/hallu_smiles/shared/
├── models/qwen-qwen2-5-7b-instruct/
│   ├── active-model.json
│   └── <HF commit SHA>/
│       ├── model-manifest.json
│       ├── .hallu_smiles_model_ready
│       └── *.safetensors
└── ragtruth/
    ├── source_info.jsonl
    ├── response.jsonl
    └── ragtruth-manifest.json
```

Один ответственный человек выполняет это из Jupyter на `c1.4`, а не из Job:

```bash
export DS_SHARED_ROOT="$DS_PROJECT_HOME/hallu_smiles/shared"
python scripts/stage_datasphere_shared_assets.py --shared-root "$DS_SHARED_ROOT" \
  --model-id Qwen/Qwen2.5-7B-Instruct
```

Qwen публичен и для него `HF_TOKEN` не нужен. Скрипт сам разрешает точный HF
commit, пишет manifest и создаёт `ready`-маркер только после полной загрузки.
Для gated-модели токен допускается только в одноразовой staging-сессии; GPU Jobs
не нуждаются в токене.

**Никому не разрешается** удалять, перезаписывать или класть рабочие кэши в
`shared/models` и `shared/ragtruth`. Jobs монтируют Project storage для чтения;
именно поэтому эта граница исключает гонки между ветками.

## Как запустить ветку

1. Убедитесь, что нужный commit отправлен в GitHub. Job скачивает публичный
   репозиторий и проверяет, что checkout ровно равен указанному SHA.
2. Сначала создайте дешёвый CPU preflight:

   ```bash
   COMMIT="$(git rev-parse HEAD)"
   RUN_ID="preflight-$(date -u +%Y%m%d-%H%M%S)"
   python scripts/render_datasphere_job.py --kind preflight \
     --commit "$COMMIT" --model-id Qwen/Qwen2.5-7B-Instruct --run-id "$RUN_ID" \
     --output datasphere/jobs/rendered/preflight.yaml
   datasphere project job execute -p <PROJECT_ID> \
     -c datasphere/jobs/rendered/preflight.yaml
   ```

   В `preflight.json` должны быть `status: "ready"`, точный revision модели и
   ненулевой `model_bytes_checked`.

3. Создайте и отправьте GPU Job. `RUN_ID` обязан включать ветку и commit,
   например `new-metrics-f529e8c-20260716`.

   ```bash
   python scripts/render_datasphere_job.py --kind qa-pilot-g1 \
     --commit "$COMMIT" --model-id Qwen/Qwen2.5-7B-Instruct --run-id "$RUN_ID" \
     --output datasphere/jobs/rendered/qa-pilot.yaml
   datasphere project job execute -p <PROJECT_ID> \
     -c datasphere/jobs/rendered/qa-pilot.yaml
   ```

GPU Job использует только `g1.1` (V100 32 GB), FP16 и один vLLM-процесс. Он
выполняет **строго по порядку** strict baseline, затем support-вариант, повторно
используя один 20-QA manifest и KG cache в собственном output. GPU-таймаут —
3 часа, то есть не более 777 600 units плюс максимум 60 секунд на graceful stop.

## Где смотреть и что забирать

- В веб-интерфейсе: **DataSphere Jobs → Launch history**. `Queued` не расходует
  GPU units; `Running` должен вскоре дать healthcheck vLLM и рост GPU utilization.
- В CLI: `datasphere project job attach --id <JOB_ID>` возобновляет поток логов,
  если соединение прервалось.
- В outputs Job: архив `preflight-<RUN_ID>.tar.gz` либо
  `qa-pilot-<RUN_ID>.tar.gz`. Внутри GPU-архива находятся
  `shared-assets-preflight.json`, `vllm.log`, `gpu.csv`, `run_metadata.json`,
  `qa_pilot_manifest.json`, `strict/`, `support/`, `comparison.json` и
  `comparison.md`.
- После завершения: `datasphere project job download-files --id <JOB_ID>`,
  затем распакуйте архив локально. Не держите GPU Job включённым для чтения
  отчётов.

Проверяйте `gpu.csv` и `run_metadata.json`: при `Running` без vLLM healthcheck
или без полезной GPU activity Job нужно отменить. Встроенное мобильное приложение
Yandex Cloud годится для общих статусов и push-уведомлений; точные Job-логи и
артефакты доступны в браузере и CLI.

## Правила для параллельной работы

- Один Job — один commit — один `RUN_ID`; никогда не переиспользуйте output
  другого эксперимента как рабочий каталог.
- Не передавайте `HF_TOKEN`, API-ключи и локальные `.env` в Git, YAML или логи.
- Обычные Jobs должны использовать только `MODEL_PATH`, найденный через
  `active-model.json`; скачивание модели в GPU Job — ошибка конфигурации.
- Перед изменением общего загрузчика или модели согласуйте это с командой. Новая
  ревизия становится активной только после полного staging и нового preflight.
