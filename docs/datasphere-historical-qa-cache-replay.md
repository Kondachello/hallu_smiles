# Проверка повторного использования исторического кэша 100-QA

Этот запуск доказывает ровно одно: фреймворк берёт три уже извлечённых графа
знаний (`context`, `query`, `response`) из исторического 100-QA хранилища и
передаёт один и тот же граф ответа HalluGraph и GraphEval. Это не новый
эксперимент качества и не извлечение графов.

## Что запускается

Шаблон Job: `datasphere/jobs/historical-qa-cache-replay.template.yaml`.
Точка запуска в контейнере:
`scripts/run_datasphere_historical_qa_cache_replay.sh`.

Перед работой оболочка запуска:

1. проверяет закреплённый CPU runtime и локальную модель HHEM;
2. читает `datasphere/historical_kg_cache_lineages.json` и
   `checkpoint-identity.json` в Project storage;
3. выбирает единственный каталог 100-QA, чья версия кода, параметры
   извлечения, fingerprint runtime и схема 80/20/5 совпадают с реестром;
4. восстанавливает тот же детерминированный набор из 100 RAGTruth QA записей;
5. вычисляет ключи кэша для всех трёх ролей, выбирает первую по стабильному
   порядку запись, у которой есть все три совместимых графа;
6. запускает HalluGraph и GraphEval в `cache_only`.

При пропуске графа выбранная запись не подменяется генерацией: запуск завершается
ошибкой. При пропуске у другой из 100 записей это не мешает выбрать первую полную
запись; отчёт сохраняет покрытие всех 100 записей.

## Гарантии стоимости и изоляции

- Только CPU `c1.4`; GPU в YAML отсутствует.
- Job получает Project secret `HALLU_GATEWAY_API_KEY` только для одного
  аутентифицированного HTTP-запроса к `/v1/hallu/manifest`. Это метаданные о версии
  gateway, а не запрос к LLM: текст RAGTruth не передаётся, извлечение не вызывается.
  Хэш manifest обязан совпасть с исторической линией; иначе Job завершается до lookup.
- В режиме `cache_only` KGGen backend не может быть сконструирован. Отчёт требует
  `kggen_api_calls = 0` и `grapheval_extractor_calls = 0`.
- HHEM — закреплённая локальная CPU модель GraphEval, не LLM и не сетевой вызов.
- В детекторы передаётся только `instances.no_gold.jsonl`; поля разметки и качества
  не попадают в этот файл.
- Исторический каталог монтируется только для чтения; текущий служебный каталог
  находится в рабочем storage Job и не меняет исторический кэш.

## Что искать в архиве Job

В `historical-qa-cache-replay-<run-id>.tar.gz` находятся:

- `historical-lineage.json` — выбранный checkpoint и его идентичность;
- `historical-cache-runtime-identity.json` — fingerprint, участвующий в ключе;
- `historical-cache-replay/historical_qa_cache_replay_report.json` — главный
  отчёт с выбранным `response_id`, статусами обоих методов и нулевыми счётчиками;
- `historical-cache-replay/reports/historical_cache_coverage.json` — все
  проверенные ключи и источники графов;
- `historical-cache-replay/shared_graphs/graph_index.jsonl` и
  `cache/cache_resolution.jsonl` — доказательство источника
  `historical_100qa` для трёх графов;
- `historical-cache-replay/prediction_seal.json` и `run_manifest.json` —
  контрольные суммы и итоговый статус;
- `replay.stdout.log` и `replay.stderr.log` — диагностические журналы.

Успешным считается только архив, где оба метода имеют `status=ok`, валиден
`prediction_seal`, все использованные источники равны `historical_100qa`, а оба
счётчика извлечения равны нулю.

## Отправка после зелёного CI

До прохождения CI Job не отправляется. После явного подтверждения пользователя
из PowerShell используется одинарная команда (Git Bash вызывается внутри неё):

```powershell
$env:YC_AUTH="yc"; $env:PYTHON_BIN="python"; $env:PATH="$env:USERPROFILE\yandex-cloud\bin;$env:PATH"; & "C:\Program Files\Git\bin\bash.exe" -lc "cd /c/Users/Kolya/Desktop/SMILES/HaluVSGraph_Eval/hallu_smiles && source .venv-datasphere/Scripts/activate && bash scripts/submit_datasphere_historical_qa_cache_replay.sh --project-id bt1i64odluitglbaj5st --branch codex/experiment-framework-spec --run-id historical-cache-YYYYMMDD --gateway-url https://hallu-vertex-gateway-453887629111.europe-west4.run.app"
```

`gateway-url` в этой команде — только часть исторического ключа кэша. Скрипт не
подключается к нему. Не меняйте URL, commit или runtime image: это изменит ключ и
превратит корректный исторический кэш в пропуск.

Связанный общий контракт: `docs/graph-cache-reuse.md`. Источник исторического
контекста: `docs/vertex-100qa-hypothesis-report.md`.
