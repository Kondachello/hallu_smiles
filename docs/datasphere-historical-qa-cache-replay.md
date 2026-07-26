# Historical QA cache-only replay (100-QA и 750-QA)

Этот запуск доказывает ровно одно: фреймворк берёт уже извлечённые графы знаний
(`context`, `query`, `response`) из исторического QA-кэша на Project storage и
передаёт их HalluGraph и GraphEval **без единого вызова LLM** (`cache_only`).
Это не новый эксперимент качества и не извлечение графов.

Свежий полный прогон и метрики: `docs/vertex-750qa-cache-replay-results.md`.

---

## 1. TL;DR — как запустить

Образ рантайма собирается GitHub Actions на каждый коммит. **Сначала запушьте код
и дождитесь зелёного CI-билда образа**, иначе submit будет висеть, ожидая образ.

```bash
# Git Bash. Токен ставим заранее, иначе datasphere CLI уйдёт в браузерный OAuth и подвиснет.
export PATH="/c/Program Files/Git/bin:$HOME/yandex-cloud/bin:$PATH"
export YC_AUTH=yc
export YC_TOKEN="$(yc iam create-token)"
export TEMP="D:\\tmp-datasphere"; export TMP="D:\\tmp-datasphere"; export TMPDIR="/d/tmp-datasphere"
cd /d/RagTruth/hallu_smiles
set -a; source .env; set +a          # HALLU_GATEWAY_URL и т.п.

RUN_ID="qa750-$(date -u +%Y%m%d-%H%M%S)"
PYTHON_BIN=python bash scripts/submit_datasphere_historical_qa_cache_replay.sh \
  --project-id bt1i64odluitglbaj5st \
  --run-id "$RUN_ID" \
  --gateway-url "$HALLU_GATEWAY_URL" \
  --qa-sample-size 750 \          # ОБЯЗАТЕЛЬНО = размеру целевого чекпойнта
  --replay-count all \            # реплеить все полностью закэшированные записи
  --commit "$(git rev-parse HEAD)" \
  --timeout-seconds 43200
```

Быстрая проверка сквозного пайплайна (не качество): `--qa-sample-size 100
--replay-count 5`.

### Флаги, которые задают, к какому кэшу подключаемся

| Флаг | Смысл | Частая ошибка |
|---|---|---|
| `--qa-sample-size N` | Размер QA-выборки И имя чекпойнта `qa-N-test-...`. **Должен совпадать с размером кэша.** | Дефолт `100`. Если оставить дефолт при кэше на 750 → материализуется другой набор текстов → все ключи мимо → `available=0`. |
| `--replay-count all` | Реплеить каждую запись, у которой все 3 роли в кэше. | Жёсткое число `750` упадёт, если хоть одна запись неполна. |
| `--replay-count N` | Реплеить ровно N полных записей (детерминированный сэмпл). | `N` > числа полных записей → ошибка `available<requested`. |
| `--gateway-url`, `--commit` | Часть ключа кэша (fingerprint, версия кода). Скрипт к gateway для инференса НЕ ходит. | Смена URL/commit/образа меняет ключ и превращает валидный кэш в промах. |
| `--diagnostic-only` | Read-only: только сверить ключи с файлами кэша, ничего не считать. | — |

---

## 2. Как устроено подключение к кэшу (это было самое сложное)

Оболочка запуска `scripts/run_datasphere_historical_qa_cache_replay.sh` внутри
Job делает discovery в три шага и **собирает цепочку из двух read-источников**.

### Модель хранения (два класса артефактов)

1. **Persistent cache на Project storage** — то, что переиспользуется:
   ```
   checkpoints/vertex-qa/
     qa-750-test-150-cv-5/
       baseline-v1-<gateway-manifest-sha>/       # writable primary кэш этого прогона
         kg/  verdicts/
       support-critical/
         support-critical-v1-<gateway-manifest-sha>/
           kg/ critical_claims/ critical_coverage/ critical_verdicts/
           checkpoint-identity.json
     qa-100-test-20-cv-5/
       <commit>-<gateway-manifest-sha>/kg/        # исторический 100-QA кэш (lineage source)
   ```
2. **Артефакт конкретного Job** (в его `RUN_ROOT`, попадает в tar.gz): `run_metadata.json`,
   `usage-counts.json`, отчёты, `raw_predictions.jsonl` и т.д. **`run_metadata.json`
   живёт здесь, а НЕ в `checkpoints/**`** — его отсутствие в кэше это норма и НЕ
   признак незавершённости.

`checkpoint-identity.json` — тоже НЕ маркер завершения: он пишется атомарно в
начале Job, чтобы закрепить совместимость namespace (gateway manifest, split
`750 = 600 train + 150 test`, CV=5, протокол кэша). Отсутствие его прямо в
`qa-750-test-150-cv-5/` нормально — это родительская папка; файл лежит глубже, в
`.../support-critical/support-critical-v1-<hash>/`.

Каждая валидная запись пишется сразу, контент-адресно и атомарно (`os.replace`).
Если Job упадёт, готовые записи остаются и переиспользуются следующим Job с тем
же protocol/gateway hash. Поэтому «нет итоговых метрик» ≠ «кэш не готов».

### Цепочка чтения (read-through)

- **primary** = `baseline-v1-<sha>/kg` целевого размера (для 750-прогона содержит
  графы «новых» ~650 записей);
- **secondary** = `qa-100-test-20-cv-5/<...>/kg` — read-through для 100 записей,
  общих у 750- и 100-выборок; их графы физически лежат в 100-QA кэше и в baseline
  не дублируются.

Скрипт подключает secondary автоматически в режиме `direct` (когда есть свежий
`baseline-v1` для запрошенного размера). В `graph_sources` итогового отчёта
должны быть **оба** id: `historical_100qa` и `historical_lineage_0`.

### Ключ кэша и legacy-схема

Ключи 750-кэша посчитаны схемой `kggen-v11-pre-length-retry` (legacy), а не
current. Источник читается с `cache_key_compatibility=(V11_PRE_LENGTH_RETRY,)`,
поэтому попадания идут по legacy-ключу. Fingerprint берётся из lineage
(`vertex-gateway:9ba169...`) как **явный override**, а не свежевычисленный — это
та идентичность, которой кэш реально записан.

---

## 3. Диагностика перед реальным прогоном (дёшево, read-only)

`--diagnostic-only` запускает `scripts/diagnose_historical_cache_key_mismatch.py`:
он вычисляет те же ключи для всех выбранных текстов и проверяет, сколько уже лежит
в кэше — по ОБОИМ каталогам (несколько `--kg-dir` подключаются автоматически).

Ключевые поля отчёта (`diagnostic-cache-key-report.json` / stdout):

- `combined_available` из `keys_probed` (ждём ≈ `3 × qa_sample_size`);
- `complete_records` — сколько записей полны по всем 3 ролям (столько отреплеит `all`);
- `probed_kg_dirs[].records_or_roles_owned` — вклад каждого источника (secondary
  должен дать ~300; если 0 — read-through не подключился);
- `missing_sources` / `missing[]` — какие записи ещё не в кэше (их дописывает
  поздний extraction-run).

Пример последнего снимка 750-QA: `combined_available=2247/2250`,
`complete_records=749`, не хватает 1 записи (`17712`/source `12448`).

---

## 4. Что искать в архиве Job и критерий успеха

В `historical-qa-cache-replay-<run-id>.tar.gz`:

- `historical-cache-replay/reports/historical_qa_cache_replay_report.json` —
  главный отчёт: `replay_count`, `detector_statuses`, `detector_status_counts`,
  нулевые счётчики, `graph_sources`;
- `.../reports/historical_cache_coverage.json` — проверенные ключи/источники;
- `.../shared_graphs/graph_index.jsonl`, `.../cache/cache_resolution.jsonl` —
  доказательство источника каждого графа;
- `.../reports/progress.jsonl` — по-записный прогресс (пишется инкрементально с
  flush; переживает падение — из него можно восстановить скоры);
- `.../prediction_seal.json`, `run_manifest.json` — контрольные суммы и статус.

**Успех** = все выбранные записи имеют статус `ok` **или** `empty_graph`,
`validation.valid=true`, `kggen_api_calls=0`, `gateway_llm_calls=0`,
`grapheval_extractor_calls=0`, а `graph_sources ⊆ {configured}`.

> `empty_graph` — легитимный не-LLM исход (пустой/невалидный граф ответа →
> `raw_score=None`, см. `graph_eval/DEVIATIONS.md` §9). Инвариант считает его
> проходным; на статус детектора он не влияет.

---

## 5. Гарантии стоимости и изоляции

- Только CPU `c1.4`; GPU в YAML отсутствует.
- Project secret `HALLU_GATEWAY_API_KEY` используется только для одного HTTP к
  `/v1/hallu/manifest` (метаданные версии gateway, не запрос к LLM; текст RAGTruth
  не передаётся). Хэш manifest обязан совпасть с исторической линией.
- В `cache_only` KGGen backend не конструируется; промах = ошибка, а не инференс.
- HHEM — закреплённая локальная CPU-модель GraphEval, не LLM и не сетевой вызов.
- В детекторы идёт только `instances.no_gold.jsonl` (без разметки/качества).
- Исторический кэш монтируется на чтение; служебный кэш Job отдельный.

---

## 6. Типовые причины `available=0` / падений (чек-лист)

1. **Не передан `--qa-sample-size`** → материализуется дефолтная выборка 100 →
   набор текстов не тот → все промахи. Передавайте размер целевого чекпойнта.
2. **Secondary-источник не подключился** (`records_or_roles_owned=0` у lineage kg)
   → 100 общих записей мимо. Проверьте, что режим `direct` и lineage kg существует.
3. **Жёсткий `--replay-count N` больше числа полных записей** → `available<requested`.
   Используйте `--replay-count all`.
4. **Изменили `--gateway-url`/`--commit`/образ** → сменился ключ → тотальный промах.
5. **submit висит** → не собран CI-образ для коммита ИЛИ не задан `YC_TOKEN`
   (уходит в OAuth). Дождитесь билда, экспортируйте свежий токен.
6. **Падение в самом конце с `invariants failed`** до коммита `566f5a8` — из-за
   `empty_graph`. На свежем коде проходит.

---

Связанные документы: `docs/graph-cache-reuse.md` (общий контракт кэша),
`docs/vertex-100qa-hypothesis-report.md` (источник исторического контекста),
`docs/vertex-750qa-cache-replay-results.md` (метрики последнего прогона).
