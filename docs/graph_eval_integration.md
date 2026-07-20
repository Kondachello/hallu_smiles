# GraphEval + HalluGraph: хендофф для experiment framework

Этот документ — для того, кто делает **третий фреймворк (experiment runner)**. Здесь
описано, что уже собрано, как оно устроено и как этим пользоваться, чтобы подключить
оба детектора единообразно и построить парное сравнение на RAGTruth.

Всё, что ниже, лежит в ветке **`sasha`** (от `origin/new-metrics`). Коммиты Stage 1–5:
`e7f54f9`, `f1183a2`, `c041d70`, `646f964`, `5814894`.

---

## 0. Оглавление
1. [Архитектура и разделение ролей](#1-архитектура-и-разделение-ролей)
2. [Быстрый старт (offline, 2 минуты)](#2-быстрый-старт-offline-2-минуты)
3. [Общий контракт детекторов](#3-общий-контракт-детекторов)
4. [GraphEval: как устроен и как звать](#4-grapheval-как-устроен-и-как-звать)
5. [HalluGraph adapter: как звать и паритет](#5-hallugraph-adapter-как-звать-и-паритет)
6. [Как вызывать оба детектора единообразно](#6-как-вызывать-оба-детектора-единообразно)
7. [Кэш и воспроизводимость](#7-кэш-и-воспроизводимость)
8. [empty_graph / failed — обязательная обработка](#8-empty_graph--failed--обязательная-обработка)
9. [Что должен сделать experiment framework](#9-что-должен-сделать-experiment-framework)
10. [Тесты](#10-тесты)
11. [Границы и запреты](#11-границы-и-запреты)
12. [Открытые пункты перед live-прогоном](#12-открытые-пункты-перед-live-прогоном)
13. [Карта файлов](#13-карта-файлов)

---

## 1. Архитектура и разделение ролей

Три независимых фреймворка:

| # | Фреймворк | Где | Кто делает |
|---|---|---|---|
| 1 | **HalluGraph** (EG/RP/CFI, 3 графа) | корень репо: `src/`, `run.py` | уже готово |
| 2 | **GraphEval** (граф из ответа + NLI каждого триплета) | `graph_eval/` | готово (Stage 1–4) |
| — | **Adapter HalluGraph → общий контракт** | `detector_adapters/` | готово (Stage 5) |
| 3 | **Experiment runner** (RAGTruth, split, tuning, метрики, отчёты) | будущий `experiments/` | **ты** |

Правило зависимостей: `graph_eval` **не** импортирует HalluGraph; HalluGraph (`run.py`/`src/`)
**не тронут** и не импортирует ничего нового; `detector_adapters/` — интеграционный слой,
который импортирует общий контракт из `graph_eval.types` и существующий HalluGraph. Твой
experiment framework импортирует **оба детектора через один контракт** и сам занимается
данными/порогами/метриками. Детекторы **никогда** не видят gold-меток.

---

## 2. Быстрый старт (offline, 2 минуты)

Оба детектора работают полностью офлайн на fake-бэкендах (без torch/openai/сети) — этого
достаточно, чтобы собрать и протестировать твой оркестратор. Реальные HHEM/gateway
включаются конфигом и запускаются в DataSphere.

```bash
# из корня репо
export PYTHONPATH="$PWD/graph_eval/src"     # или: pip install -e graph_eval

# GraphEval через CLI: no-gold JSONL -> predictions JSONL
printf '%s\n' \
 '{"response_id":"r1","source_id":"s1","context":"Paris is the capital of France.","response":"Paris is the capital of France.","query":"capital?"}' \
 > /tmp/in.jsonl
.venv/bin/python -m graph_eval.cli predict --input /tmp/in.jsonl --output /tmp/preds.jsonl
cat /tmp/preds.jsonl
```

Программно (тоже offline, fake):

```python
from graph_eval import GraphEvalDetector, DetectionInput
from graph_eval.extraction.fake import FakeExtractor
from graph_eval.nli.fake import FakeNLI

det = GraphEvalDetector(FakeExtractor(), FakeNLI())
res = det.predict(DetectionInput(
    response_id="r1", source_id="s1",
    context="Paris is the capital of France.",
    response="Berlin is the capital of France.",
    query="capital of France?"))
print(res.status, res.raw_score, res.flagged_unit_ids)   # ok <score> (...)
```

---

## 3. Общий контракт детекторов

Определён один раз в `graph_eval/types.py`. Оба детектора принимают `DetectionInput` и
возвращают `DetectionResult`. Это **вся** твоя точка интеграции.

### `DetectionInput` (frozen dataclass)
| поле | тип | смысл |
|---|---|---|
| `response_id` | `str` | id ответа |
| `source_id` | `str` | id источника (для source-grouped split и bootstrap) |
| `context` | `str` | grounding-контекст (единственное evidence) |
| `response` | `str` | проверяемый ответ |
| `query` | `str \| None` | вопрос; сохраняется, но **не** используется как evidence |
| `metadata` | `Mapping` | только неголдовые поля (task, gen_model, …) |

Gold-полей тут нет **по построению**. Ты присоединяешь gold к предсказанию **после**
вызова детектора, у себя.

### `DetectionResult` (frozen dataclass)
| поле | тип | смысл |
|---|---|---|
| `response_id`, `source_id` | `str` | эхо входа |
| `method` | `str` | `"grapheval"` или `"hallugraph"` |
| `raw_score` | `float \| None` | **больше = вероятнее галлюцинация**; `None` если не `ok` |
| `components` | `Mapping` | детали метода (см. ниже) — сюда смотри для срезов/аудита |
| `flagged_unit_ids` | `tuple[str, …]` | локализация: подозрительные триплеты/сущности/рёбра |
| `status` | `str` | `ok` \| `empty_graph` \| `failed` |
| `failure` | `Mapping \| None` | причина при `failed` (stage + error) |
| `usage` | `Mapping` | calls / cache hits / latency / tokens |
| `artifact_refs` | `Mapping` | ссылки на артефакты (пусто по умолчанию) |

**Инварианты (гарантируются кодом, можешь на них опираться):**
- `raw_score` имеет единое направление: **выше = больше галлюцинация**, у обоих методов.
- если `status != "ok"` → `raw_score is None` (пустой/битый граф **не** маскируется числом 0/1).
- `status == "failed"` → `failure` непустой; транспортная ошибка **никогда** не превращается
  в «галлюцинацию» (это отдельное состояние).
- детектор не читает gold и не знает про train/test/threshold.

---

## 4. GraphEval: как устроен и как звать

### Алгоритм
1. Экстрактор строит триплеты **только из `response`** (контекст/query ему не передаются).
2. Каждый триплет вербализуется в `"<subject> <relation> <object>."`.
3. NLI (HHEM) получает `premise = context`, `hypothesis = вербализация` → `p_consistent`.
4. `p_unsupported = 1 − p_consistent`; ответный score `H = max_i p_unsupported_i`.
5. Paper-решение: галлюцинация ⇔ `H > 0.5` (кладётся в `components.paper_threshold_decision`).
   Твой train-tuned порог применяешь ты, отдельно — на `raw_score`.

`components` у GraphEval:
```
triples: [ {triple_id, raw_subject, raw_relation, raw_object,
            verbalized_hypothesis, p_consistent, p_unsupported,
            flagged_at_paper_threshold}, ... ]
n_triples_total / n_triples_valid / n_triples_invalid
aggregation ("max_unsupported"), verbalizer_version, paper_threshold,
paper_threshold_decision (bool | None)
```
`flagged_unit_ids` = id триплетов с `p_unsupported > paper_threshold`.

### Конфиг
`GraphEvalConfig` (в `graph_eval/config.py`), собирается из dict через `from_dict(...)`:

```yaml
extractor:
  backend: gateway            # fake | gateway | vllm(зарезервировано)
  model: openai/gemini-2.5-flash
  prompt_profile: grapheval_appendix_a_v1
  output_mode: paper_prompt   # paper_prompt | structured_json
  temperature: 0.0
  max_tokens: 2048
  max_retries: 5              # транспортные ретраи (429/5xx/timeout)
  max_repairs: 2              # corrective re-prompt при truncation/невалидном JSON
  api_base_env: HALLU_GATEWAY_URL
  api_key_env: HALLU_GATEWAY_API_KEY   # только имя переменной, НЕ значение
  cache_namespace: grapheval/extraction/v1
nli:
  backend: hhem               # fake | hhem
  model: vectara/hallucination_evaluation_model
  revision: REQUIRED_EXACT_COMMIT_SHA  # ОБЯЗАТЕЛЬНО точный HF-commit (конфиг отвергает main)
  model_label: hhem-2.1-open
  device: cpu
  dtype: float32
  batch_size: 8
  evidence_policy: full_context_native
  aggregation: max_unsupported
  paper_threshold: 0.5
  cache_namespace: grapheval/nli/v1
cache_dir: .cache/graph_eval
cache_only: false             # true = warm-replay: 0 сетевых вызовов, miss = ошибка
```
Пустой конфиг → всё `fake` (offline). `config.validate()` кидает ошибку, если у `hhem`
не запинена revision, если порог вне [0,1] или backend неизвестен.

### Сборка через factory (рекомендуется)
```python
from graph_eval import from_dict, GraphEvalDetector
from graph_eval.factory import build_extractor, build_nli

cfg = from_dict(cfg_dict)                       # твой YAML -> dict -> config
detector = GraphEvalDetector(
    build_extractor(cfg, manifest_sha256=GATEWAY_MANIFEST_SHA256),  # обёрнут в кэш
    build_nli(cfg),                                                  # обёрнут в кэш
    paper_threshold=cfg.nli.paper_threshold,
    aggregation=cfg.nli.aggregation,
)
res = detector.predict(item)
```
- `build_extractor(cfg, *, client=None, manifest_sha256=None, cache=True)` — `fake`→FakeExtractor,
  `gateway`→GatewayExtractor (клиент openai создаётся лениво; можно инжектить свой для тестов).
  `manifest_sha256` входит в ключ кэша (см. §7) — передавай хеш аутентифицированного
  gateway-манифеста.
- `build_nli(cfg, *, cache=True)` — `fake`→FakeNLI, `hhem`→HHEMNLIModel (torch грузится лениво).

### CLI (standalone, resume-safe)
```bash
python -m graph_eval.cli predict \
  --config config.yaml \
  --input instances.no_gold.jsonl \
  --output predictions.jsonl \
  --manifest-sha256 <gateway_manifest_sha256> \
  [--resume] [--limit N]
```
Вход — JSONL без gold (поля: `response_id, source_id, context, response, query?, metadata?`;
любые gold-подобные поля игнорируются). Выход — по строке на предсказание:
`prediction_record` = `{response_id, source_id, method, raw_score, status,
flagged_unit_ids, components, failure, usage, query_present}`. `--resume` пропускает
`response_id`, уже присутствующие в выходе (можно прерывать и продолжать; кэши доделают
остальное). Console-script: `graph-eval predict ...` (после `pip install -e graph_eval`).

---

## 5. HalluGraph adapter: как звать и паритет

`detector_adapters/hallugraph_adapter.py` оборачивает **существующий** HalluGraph в тот же
контракт, **переиспользуя** `run.build_refgraph` + `src.metrics.score_response` дословно и
читая `EG/RP/cfi_for_mode/h_for_mode` ровно как `run.build_rows`. Поэтому паритет с
`run.py` — по построению (есть тест). `run.py`/`src/` **байт-в-байт как `new-metrics`**.

```python
import run
from src.config import load_config
from src.extract import UsageLogger
from detector_adapters.hallugraph_adapter import HalluGraphAdapter

cfg = load_config("config.yaml")                 # это конфиг HalluGraph (не GraphEval!)
usage = UsageLogger("results/usage.jsonl")
fake = True                                      # True=offline (FakeKGGen+DictEmbedder)
extractor = run.get_extractor(cfg, fake, usage)  # реюз конструкторов run.py
embedder  = run.get_embedder(cfg, fake)

adapter = HalluGraphAdapter(cfg, extractor, embedder,
                            alpha=0.7, tau_e=None, tau_r=None, relation_mode="strict")
res = adapter.predict(item)      # тот же DetectionInput/DetectionResult
```

`components` у HalluGraph: `EG, RP, RP_strict, RP_support, CFI, alpha, relation_mode,
tau_e, tau_r, Vc, Ec, Vq, Eq, Va, Ea, unscorable, ref_empty, H_eg, ungrounded_entities,
unsupported_relations`. `raw_score = H = 1 − CFI`. `flagged_unit_ids` = `entity:<...>` +
`relation:<s|p|o>` для несопоставленных сущностей/рёбер. Va==0 → `empty_graph` (raw_score None).

**Важно для тюнинга (как в `run.py`):**
- **α не требует повторного вызова детектора.** `EG` и `RP_strict` лежат в `components`;
  на train можешь считать `CFI = α·EG + (1−α)·RP` и `H = 1−CFI` для любого α — экстракция
  (дорогая часть) уже в кэше. Именно так делает `run.build_rows`.
- **τ_e/τ_r влияют на матчинг (RefGraph), а не на экстракцию.** Смена τ требует повторного
  `predict` с другими `tau_e/tau_r` (графы берутся из кэша). Так делает `run.tune_joint`.
- `relation_mode`: `strict` (без verifier, LLM-free), `support`, `support-critical`
  (нужны `verifier` / `critical_pipeline`; конструируй их через `run.get_verifier` /
  `run.get_critical_pipeline`). Для первого сравнения бери `strict`.

Реальный (не fake) HalluGraph требует KGGen через gateway + SBERT-эмбеддер (torch) — это
только внутри DataSphere.

---

## 6. Как вызывать оба детектора единообразно

```python
from graph_eval import DetectionInput

def to_input(row):                       # row — твоя строка RAGTruth БЕЗ gold в детектор
    return DetectionInput(
        response_id=row["response_id"], source_id=row["source_id"],
        context=row["context_raw"], response=row["response_raw"],
        query=row.get("query_raw"),
        metadata={"task": row.get("task"), "gen_model": row.get("gen_model")},
    )

def run_detector(detector, rows):
    for row in rows:                     # ОДНИ И ТЕ ЖЕ (C,Q,A) в оба детектора
        res = detector.predict(to_input(row))
        # gold присоединяешь ЗДЕСЬ, после предсказания:
        yield {"response_id": res.response_id, "method": res.method,
               "raw_score": res.raw_score, "status": res.status,
               "components": res.components, "flagged": list(res.flagged_unit_ids),
               "usage": res.usage, "gold": row["gold_response_label"]}
```
`grapheval` и `hallugraph` дают одинаковую форму результата → сравнение парное, на одном
manifest/split. Порог применяешь к `raw_score` сам (train-tuned), paper-порог у обоих
доступен отдельно (GraphEval: `components.paper_threshold_decision`; HalluGraph: посчитай от
своего θ).

---

## 7. Кэш и воспроизводимость

GraphEval пишет **атомарный** content-addressed кэш (`os.replace` + уникальный temp),
раздельные namespaces под `cache_dir`:
```
cache/graph_eval/grapheval/extraction/v1/   # сырой вывод экстрактора по ответу
cache/graph_eval/grapheval/nli/v1/          # p_consistent по (premise, hypothesis)
```
Ключи включают то, что меняет результат:
- extraction: `prompt_profile + prompt_version + schema_version + output_mode + model +
  gateway_manifest_sha256 + temperature + max_tokens + canonical(response)`;
- nli: `nli_model + nli_model_revision + evidence_policy + verbalizer_version +
  canonical(premise, hypothesis)`.

`cache_only: true` → warm-replay: **0 сетевых вызовов**, а miss — это `CacheOnlyMissError`
(она пробрасывается наружу, а не превращается в per-item `failed`). Проверено e2e-тестом:
второй идентичный прогон даёт byte-identical предсказания и ноль вызовов. HalluGraph
использует свои кэши (`.cache/kg` + verdicts) — они не пересекаются с GraphEval namespace.

`usage` в `DetectionResult` уже разделяет model-calls и cache-hits
(`extractor_calls / extraction_cache_hits / nli_calls / nli_cache_hits`) — удобно для
cold/cached cost-профилей.

---

## 8. `empty_graph` / `failed` — обязательная обработка

Пустой граф ответа нельзя молча считать «фактическим», а сбой — «галлюцинацией»
(это требование протокола, §9 плана). Детектор возвращает это отдельными состояниями;
**решение — на твоей стороне**:
- `status == "empty_graph"` (`raw_score is None`): выбери заранее **≥2 политики** и
  отчитайся раздельно — (а) primary predefined policy для бинарного решения; (б) sensitivity
  с исключением unscorable и явным знаменателем.
- `status == "failed"` (`raw_score is None`, есть `failure`): это отдельное состояние, не
  score 0/1. Логируй `failure.stage`/`error`, считай отдельно.
- В отчёте всегда показывай число обработанных / исключённых / упавших по методу и причине.

---

## 9. Что должен сделать experiment framework

Твоя зона (Stage 6 плана + COMPARISON_PROTOCOL.md):
1. RAGTruth adapter: загрузка `(context, query, response)` без изменений; сохранить
   неизменяемый input manifest и хеши.
2. **Изоляция gold**: убрать все gold-поля до вызова детектора (передавать только 6 полей
   `DetectionInput`). Присоединять gold только к предсказаниям.
3. **Source-grouped split**: ответы с одним `source_id` не текут между train/dev/test.
4. Вызвать **оба** детектора на одних и тех же примерах (см. §6).
5. **Train-only tuning**: порог θ (и для HalluGraph α, τ_e, τ_r) выбирать только на train/dev,
   затем заморозить. GraphEval: тюнишь только θ на `raw_score`. HalluGraph: α/τ по §5.
6. Один финальный frozen test-прогон.
7. Метрики: AUROC, AUPRC, balanced accuracy, P/R/F1, calibration; paired bootstrap по
   `source_id`, McNemar; обязательные срезы (task, gen_model, длина контекста, плотность
   графа, `due_to_null`/`implicit_true`).
8. Span-локализация из `flagged_unit_ids` (IoU с gold spans), error audit.
9. Треки сравнения: **A** — faithful (каждый метод в своей авторской схеме) и **B** —
   controlled (одинаковый граф ответа/препроцессинг, различается только verifier).
10. Общий archive: commit, config/model/prompt/manifest hashes, usage, failed-items, results.

Детекторы уже дают тебе всё нужное сырьё (`raw_score`, `components`, `flagged_unit_ids`,
`usage`, `status`). Ничего в `graph_eval/` или `src/` тебе менять не нужно.

---

## 10. Тесты

```bash
export PYTHONPATH="$PWD/graph_eval/src"
.venv/bin/python -m pytest graph_eval/tests detector_adapters/tests -q --noconftest
# 62 passed — всё офлайн (fake-бэкенды), без torch/openai/сети
```
Отдельные наборы: `graph_eval/tests` (ядро, HHEM-адаптер, кэши, gateway, CLI, e2e
cache-only replay) и `detector_adapters/tests` (паритет HalluGraph-адаптера с
`run.build_rows`). Fake-бэкенды детерминированы — используй их, чтобы тестировать свой
оркестратор без облака.

---

## 11. Границы и запреты
- **`HALLU_GATEWAY_API_KEY`** нигде не печатать/коммитить; в конфиге — только имя переменной.
  DataSphere подставляет значение в контейнере. `GOOGLE_APPLICATION_CREDENTIALS` не использовать,
  gateway URL не подменять.
- Детекторам **не** передавать gold (метки, spans, `due_to_null`, `implicit_true`, quality).
- Параметры и пороги подбирать **только** на train/dev, не по held-out test.
- Пустые графы / parse-ошибки / refusal / truncated не выбрасывать молча — отдельная строка и
  заранее заданное правило.

---

## 12. Открытые пункты перед live-прогоном
- **Запинить точный HF-commit HHEM** в `nli.revision` (конфиг отвергает `main`/незапиненное).
  Веса стейджатся/включаются в воспроизводимый runtime; HHEM гоняется локально в DataSphere Job.
- **Собрать DataSphere-job для GraphEval**, переиспользуя существующий gateway/CI (submitter,
  immutable image по commit, gateway manifest в cache identity, cache-only replay proof). Это
  частично пересекается с твоим Stage 6 — согласуем, чтобы не плодить второй несовместимый путь.
- Заморозить перед финальным test: extractor mode (`paper_prompt` vs `structured_json`),
  prompt/verbalizer версии, decoding/retry/repair, empty/failure policy, NLI batch/device,
  paper vs train-tuned threshold, срезы.

---

## 13. Карта файлов
```
graph_eval/                         # фреймворк #2 (устанавливаемый пакет)
  pyproject.toml  README.md  DEVIATIONS.md  config.example.yaml
  src/graph_eval/
    types.py         # DetectionInput/Result, Triple, статусы  <- ОБЩИЙ КОНТРАКТ
    config.py        # GraphEvalConfig/ExtractorConfig/NLIConfig, from_dict, validate
    detector.py      # GraphEvalDetector.predict
    factory.py       # build_extractor / build_nli (backend dispatch + кэш)
    parser.py verbalize.py scoring.py cache.py usage.py artifacts.py cli.py
    extraction/  base.py fake.py gateway.py cached.py prompt.py retry.py
    nli/         base.py fake.py hhem.py cached.py
  tests/             # 59 офлайн-тестов
detector_adapters/                  # интеграционный слой
  hallugraph_adapter.py             # HalluGraph -> общий контракт (реюз run.py, паритет)
  tests/                            # 3 parity-теста
src/  run.py                        # HalluGraph (фреймворк #1) — НЕ ТРОНУТ
docs/graph_eval_integration.md      # этот файл
```

Вопросы по контракту/вызову — пиши, докручу. Всё в `graph_eval` и `detector_adapters`
покрыто офлайн-тестами, так что можно цеплять и гонять локально без облака.
