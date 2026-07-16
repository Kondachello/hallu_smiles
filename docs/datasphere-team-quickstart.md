# Быстрый старт DataSphere для команды

## База для новых задач

**Каждую новую экспериментальную ветку создаём от актуального `new-metrics`.**

```bash
git fetch origin
git switch new-metrics
git pull --ff-only origin new-metrics
git switch -c <your-branch>
```

Не начинайте от `main`, старого `sample-qa-test` или локального незапушенного
состояния: DataSphere Job клонирует только commit, опубликованный в GitHub.

## Что уже есть

- Общие Llama 3.1 8B и RAGTruth находятся в Project storage.
- GPU Job читает их read-only: модель заново не скачиваем.
- Не меняем закреплённый `vllm==0.6.3.post1`: он выбран для driver V100 `g1.1`.
- `HF_TOKEN` обычным Jobs не нужен и не передаётся.
- Общие `shared/models` и `shared/ragtruth` не редактируем и не удаляем.

## Запуск своей ветки

1. Закоммитьте и отправьте ветку в GitHub.
2. Выберите уникальный lowercase `RUN_ID`, например
   `my-experiment-a1b2c3d-20260717`.
3. Всегда начните с CPU preflight:

   ```bash
   PROJECT_ID=<PROJECT_ID>
   BRANCH=<your-branch>
   RUN_ID=preflight-<your-branch>-<date>

   PYTHON_BIN=.venv-datasphere/bin/python \
   DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
   bash scripts/submit_datasphere_job.sh \
     --kind preflight --project-id "$PROJECT_ID" --branch "$BRANCH" --run-id "$RUN_ID"
   ```

4. Запускайте `qa-pilot-g1` только после `SUCCESS` preflight.

   ```bash
   RUN_ID=<your-branch>-<short-sha>-<date>
   PYTHON_BIN=.venv-datasphere/bin/python \
   DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
   bash scripts/submit_datasphere_job.sh \
     --kind qa-pilot-g1 --project-id "$PROJECT_ID" --branch "$BRANCH" --run-id "$RUN_ID"
   ```

Helper сам проверит опубликованный commit и YAML до создания Job. Если он
завершился ошибкой, Job не был создан и units не потрачены.

## Проверка и завершение

```bash
.venv-datasphere/bin/datasphere --profile default project job get \
  --id <JOB_ID> --format json
```

- `PREPARING` — ожидание/подготовка среды.
- `EXECUTING` — Job работает.
- `SUCCESS`, `ERROR`, `CANCELLED` — GPU уже освобождён автоматически.

После terminal status скачайте архив Job и не держите VM ради чтения отчётов.
Job уже ограничен 3 часами и сам остановит vLLM при normal exit, timeout,
ошибке или отмене.

## Не делать

- Не создавать новый DataSphere Project/cloud.
- Не запускать `huggingface-cli download` или `pip install` внутри GPU Job.
- Не копировать чужие caches, manifests и outputs в свой запуск.
- Не использовать `DS_PROJECT_HOME` в JupyterLab: это переменная только Job.
- Не отправлять tokens, веса модели, `outputs/`, `.venv-datasphere/` или `.tools/` в Git.

Полная инструкция с диагностикой: [datasphere-team-runbook.md](datasphere-team-runbook.md).
