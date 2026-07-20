# Памятка команде: запуск HalluGraph через общий Cloud Run и DataSphere

Это единственный актуальный путь для реальных QA-экспериментов. DataSphere
запускает KGGen, локальную CPU-модель эмбеддингов, metrics и кэши; запросы к
Gemini идут только по HTTPS через уже развёрнутый Cloud Run gateway.

## Что уже подготовлено

| Компонент | Значение / ответственность |
|---|---|
| DataSphere project | `bt1i64odluitglbaj5st` |
| Cloud Run gateway | `https://hallu-vertex-gateway-453887629111.europe-west4.run.app` |
| Vertex region / model | `europe-west4` / `gemini-2.5-flash` |
| DataSphere secret | `HALLU_GATEWAY_API_KEY` уже создан в Project и автоматически передаётся Job как переменная окружения |
| Gateway identity | Service account Cloud Run использует ADC; приватного Google key в репозитории и DataSphere нет |

**Никогда не просите и не выводите `HALLU_GATEWAY_API_KEY`.** Агенту не нужно
знать его значение: job template не содержит секрет, а DataSphere подставляет
его только внутри контейнера. Не сохраняйте ключ в YAML, `.env`, git history,
логах или сообщениях.

## Протокол эксперимента

Для одного запуска выбирается детерминированный source-level QA manifest: по
одному response на source, поровну меток `y=0/1` в train и test. Strict,
support и support-critical используют ровно этот же manifest и те же графы.

Стандартный запуск сейчас — **100 QA**:

| Часть | Записей | Назначение |
|---|---:|---|
| Train | 80 | Только здесь выполняется stratified 5-fold CV для выбора `α`, `τ_e`, `τ_r` и `θ`. |
| Held-out test | 20 | Финальные парные метрики; test не участвует в настройке. |
| Всего | 100 | Сравнение трёх детекторов на тех же графах и том же split. |

`support-critical` — отдельная экспериментальная метрика: она не меняет
historical strict/support формулы. Она добавляет atomic claims, строгие verdicts
`entailed/unknown/unsupported/contradicted` и full-context review. Любой
`unsupported` или `contradicted` claim получает максимальный риск; параметры
агрегации выбираются только на train.

Новый runner читает прежний 100-QA KG/verdict checkpoint только в режиме
read-through. Поэтому strict/support воспроизводятся cache-only, KGGen не
получает новых запросов, а live Vertex calls допускаются только для новых
critical claim/review/verdict artifacts. Финальный replay всех трёх режимов
обязан сделать ноль API calls и создать byte-identical `metrics.csv`.

## Разовый локальный минимум

Работайте из корня репозитория и не создавайте новую ветку для запуска:

```bash
cd ~/Projects/hallu_smiles
git switch new-metrics
git pull --ff-only origin new-metrics
source .venv/bin/activate

datasphere --profile default project get --id bt1i64odluitglbaj5st
```

Команда `project get` должна показать `Online_project19_1`. Если локальный
`yc` находится вне `PATH`, добавьте только на текущую сессию:

```bash
export PATH="$HOME/yandex-cloud:$PATH"
```

Это нужно для авторизации DataSphere CLI, но **не** для Vertex и не даёт
доступа к секрету gateway.

## Получить проверенный 3-QA gate

Перед полным экспериментом submitter проверяет успешный 3-QA archive: он
подтверждает модель, gateway manifest, KGGen structured path и cache-only
replay. Утверждённый gate Job: `bt1u04j0is4cfkutqocg`.

```bash
export PROJECT_ID=bt1i64odluitglbaj5st
export GATE_DIR="$PWD/outputs/datasphere-gates/vertex-3qa-r11"
mkdir -p "$GATE_DIR"

datasphere --profile default project job download-files \
  --id bt1u04j0is4cfkutqocg --with-logs --with-diagnostics \
  --output-dir "$GATE_DIR"

export GATE_ARTIFACT="$(find "$GATE_DIR" -name 'vertex-cpu-probe-vertex-3qa-20260719-r11.tar.gz' -print -quit)"
test -n "$GATE_ARTIFACT" && test -f "$GATE_ARTIFACT"
```

Не подменяйте gate архив «похожим» файлом: submitter проверит его manifest и
требует, чтобы его source commit был предком запускаемого commit.

## Запустить 100 QA / 80–20 / 5-fold

После изменения кода сначала его нужно запушить: GitHub Actions собирает
immutable CPU image для точного commit. Submitter дождётся появления image
digest (до 30 минут по умолчанию), сверит его с commit и только затем создаст
DataSphere Job.

Если на локальной машине временно не разрешается `github.com`, но `git branch -vv` уже показывает нужный commit у `origin/new-metrics`, к команде ниже можно
добавить `--skip-origin-fetch`. Этот флаг не отключает проверку SHA: он сверяет
commit с локально сохранённой remote-ссылкой. Не используйте его, если branch
ещё не был успешно отправлен в GitHub.

```bash
export PROJECT_ID=bt1i64odluitglbaj5st
export GATEWAY_URL=https://hallu-vertex-gateway-453887629111.europe-west4.run.app
export RUN_ID="vertex-100qa-$(git rev-parse --short HEAD)-$(date -u +%Y%m%d-%H%M)"

bash scripts/submit_datasphere_vertex_qa_pilot.sh \
  --project-id "$PROJECT_ID" \
  --run-id "$RUN_ID" \
  --gateway-url "$GATEWAY_URL" \
  --gate-artifact "$GATE_ARTIFACT" \
  --qa-sample-size 100 \
  --qa-test-fraction 0.2 \
  --cv-folds 5 \
  --concurrency 1 \
  --timeout-seconds 43200
```

Имя файла submitter историческое (`*_qa_pilot.sh`), но это уже
параметризованный strict/support/support-critical runner: он не зашит на 20 QA.
Для первого critical запуска **не меняйте** размер, split, gateway или seed:
он намеренно использует прежний 100-QA checkpoint. Для другой величины
меняются `--qa-sample-size` и при необходимости `--qa-test-fraction`; оба
получившихся размера train/test должны быть положительными и чётными, чтобы
сохранить баланс меток. Например, `20` + `0.2` даёт `16/4`, а `100` + `0.2`
даёт `80/20`.

`--concurrency 1` выбран для устойчивости к Vertex capacity `429`. Не
увеличивайте его без отдельного решения: retry защищает корректность, но не
создаёт дополнительную квоту Vertex.

## Наблюдение и получение результатов

После submitter рядом с rendered YAML появляется execution JSON с `job_id`.
Проверять состояние можно без attach:

```bash
datasphere --profile default project job get --id <JOB_ID> --format json
```

После terminal status скачайте и сохраните весь результат:

```bash
export OUT="outputs/datasphere-results/$RUN_ID"
mkdir -p "$OUT"
datasphere --profile default project job download-files \
  --id <JOB_ID> --with-logs --with-diagnostics --output-dir "$OUT"
```

Полный archive называется `vertex-cpu-qa-<RUN_ID>.tar.gz`. В нём обязательны:

- `qa_manifest.json`, `runtime_config.yaml` без секрета, gateway/runtime
  manifests и `run_metadata.json`;
- `strict/`, `support/` и `support-critical/` с `metrics.csv`, `tuning.json`,
  audit JSON и пустым
  `failed_extractions.jsonl`;
- `comparison.json`, `support-critical-diagnostic.json`, `usage-counts.json`,
  hashes исторического и нового cache namespaces;
- `cache-replay/strict`, `cache-replay/support` и
  `cache-replay/support-critical` с идентичными CSV и нулевыми live API calls.

При `ERROR` сначала скачайте archive и прочитайте `qa.stderr.log` и
`qa.stdout.log`; не запускайте новый job вслепую. Повторный запуск с тем же
commit, gateway manifest и параметрами sample использует durable checkpoints,
так что уже успешные KG/verdict cache entries не запрашиваются повторно.

## Что агентам разрешено и запрещено делать

- Разрешено: менять размер QA sample, test fraction, CV folds, timeout и
  concurrency через аргументы submitter; читать status/downloaded artifacts;
  анализировать strict/support/support-critical только как парное сравнение
  одного manifest.
- Запрещено: менять Cloud Run URL на произвольный endpoint, передавать ключ в
  YAML, использовать `GOOGLE_APPLICATION_CREDENTIALS`, отключать cache-only
  proof или подбирать параметры по held-out test.
- Для нового Cloud Run revision сначала нужен новый успешный 3-QA gate: manifest
  hash намеренно не позволяет полному job молча сменить model/revision.
