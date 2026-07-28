# Typed-vertex metric pass (CFI_type over the QA graph cache)

> Commit: `d5b6808` (branch `zhenya`; not yet pushed to the shared branch).
> Related: metric origin [vertex-100qa-hypothesis-report.md](vertex-100qa-hypothesis-report.md);
> cache mechanics [datasphere-historical-qa-cache-replay.md](datasphere-historical-qa-cache-replay.md);
> typing feature [dynamic_typing_agent/docs/BEST_EFFORT_TYPING.md](../dynamic_typing_agent/docs/BEST_EFFORT_TYPING.md).

## Что это и зачем

Прошлые прогоны ранжировали ответы в основном по **рёберной** компоненте (RP) HalluGraph.
Этот проход добавляет недостающий **вершинный** сигнал, но грунтит вершины ответа не по
сходству поверхностных строк, а по **назначенному типу** (динамический агент типизации):

```
EG_type  = |{v ∈ V_A : type_match(v, V_ref)}| / |V_A|      # НОВОЕ: по типам
RP       = |{e=(s,r,o) ∈ E_A : match(s,V_ref) ∧ match(o,V_ref)}| / |E_A|   # как в HalluGraph
CFI_type = α·EG_type + (1−α)·RP
raw_score (больше = вероятнее галлюцинация) = 1 − CFI_type
```

- `type_match(v, w)` — пересечение нормализованных множеств назначенных типов (см.
  [src/typed_matching.py](../src/typed_matching.py)); зеркалит `RefGraph.match_entity`
  из [src/matching.py](../src/matching.py), но по типам, а не по S-BERT-поверхности.
- `RP` (рёбра) — исходное HalluGraph-грунтование рёбер, считается **свежо** в этом же
  прогоне тем же `RefGraph` (S-BERT-эмбеддер).
- Метрика **self-contained**: всё считается в рамках прогона, результаты других
  экспериментов (strict/support) **не читаются** — их сверяем пост-фактум.
- ROC-AUC / подбор `α, τ, θ` (5-fold CV) — как в отчётах гипотез; здесь прогон выдаёт
  per-record `raw_score` + компоненты, оценка AUC делается на них.

## Архитектура (файлы)

| Файл | Роль |
|------|------|
| [src/typed_matching.py](../src/typed_matching.py) | `TypedRefGraph` + `typed_entity_grounding` + `typed_cfi` — ядро EG_type. |
| [experiments/typed_vertex_detector.py](../experiments/typed_vertex_detector.py) | `TypedVertexDetector` (DetectorProtocol): EG_type + RP → CFI_type → `DetectionResult`. Тайпер инъектируется. |
| [experiments/typed_vertex_typer.py](../experiments/typed_vertex_typer.py) | `AgentTyper` — обёртка `DynamicTypingAgent`: `build_source_registry` (context+query) + `annotate_answer` (response) → тип на каждую вершину (gateway LLM + локальный HHEM NLI). |
| [experiments/typed_metric_pass.py](../experiments/typed_metric_pass.py) | `run_typed_metric_pass`: переиспользует резолв кэша + shared-graph провайдер, скорит записи, пишет `typed_metrics.jsonl` + `typed_metric_summary.json`. |
| [scripts/typed_metric_pass.py](../scripts/typed_metric_pass.py) | CLI-обёртка. |
| [scripts/run_datasphere_historical_qa_cache_replay.sh](../scripts/run_datasphere_historical_qa_cache_replay.sh) | ветка `TYPED_METRIC_PASS=1` после резолва кэша вызывает наш скрипт. |
| [datasphere/jobs/historical-qa-typed-metric-pass.template.yaml](../datasphere/jobs/historical-qa-typed-metric-pass.template.yaml) | шаблон DataSphere-джобы. |
| [scripts/render_datasphere_typed_metric_pass_job.py](../scripts/render_datasphere_typed_metric_pass_job.py) | рендер джобы (подставляет commit SHA, image, gateway, `--alpha`). |
| [tests/test_typed_matching.py](../tests/test_typed_matching.py), [tests/test_typed_vertex_detector.py](../tests/test_typed_vertex_detector.py) | локальные юнит-тесты (без кэша/gateway). |

## Доступ к кэшу (переиспользуется, не переписан)

Джоба гоняет **тот же** драйвер [run_datasphere_historical_qa_cache_replay.sh](../scripts/run_datasphere_historical_qa_cache_replay.sh),
который уже делает всё разрешение исторического кэша (подробно — в
[datasphere-historical-qa-cache-replay.md](datasphere-historical-qa-cache-replay.md)):
аутентифицированный gateway-манифест → `GATEWAY_MANIFEST_SHA256` → поиск целевого
checkpoint `qa-<N>-test-<t>-cv-<k>/baseline-v1-<sha>/kg` (+ цепочка lineage-kg) →
пинованный `--llm-runtime-fingerprint-override`. Мы добавили лишь **gated-ветку**:

```bash
if [[ "${TYPED_METRIC_PASS:-0}" == "1" ]]; then
  export HALLU_TYPING_MODEL=... HALLU_GATEWAY_URL=$RECORDED_GATEWAY_URL \
         HALLU_HHEM_MODEL_PATH=/opt/hallu/models/hhem-2.1-open
  "$CLIENT_PYTHON" scripts/typed_metric_pass.py \
    --hallugraph-config "$RUN_ROOT/historical-cache-runtime.yaml" \
    --grapheval-config graph_eval/config.datasphere.one-instance.shared-kggen.live.yaml \
    --typing-config dynamic_typing_agent/config/live-gateway-hhem.yaml \
    --historical-cache-root "$HISTORICAL_CACHE_ROOT" [--additional-cache-root "$LINEAGE_KG_DIR"] ...
fi
```

Графы (context/query/response) читаются **cache-only** — новых вызовов экстрактора нет.
LLM-вызовы делает **только типизация** (это новая, разрешённая стоимость прогона).

## Как запустить на DataSphere

Порядок обязателен: **сначала код на GitHub, потом джоба** (джоба клонирует коммит по SHA).

```bash
# 1) запушить коммит в ветку (пока держим на zhenya; в общую — когда отработает)
git push origin zhenya:codex/dynamic-entity-typing        # (см. раздел «Ветка»)

# 2) отрендерить джобу на ЭТОТ commit SHA
python scripts/render_datasphere_typed_metric_pass_job.py \
  --commit "$(git rev-parse HEAD)" \
  --run-id typed-750all-$(date +%Y%m%d-%H%M%S) \
  --gateway-url "https://hallu-vertex-gateway-453887629111.europe-west4.run.app" \
  --docker-image "ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu@sha256:844c657d83de95feefe76ae79911cd5077d9038a753d92c43888666f524dc9db" \
  --qa-sample-size 750 --replay-count all --alpha 0.5 \
  --output datasphere/jobs/rendered/typed-750all-<ts>.yaml

# 3) сабмит в DataSphere (как для replay — см. submit_datasphere_historical_qa_cache_replay.sh)
datasphere project job execute -p <PROJECT_ID> -c datasphere/jobs/rendered/typed-750all-<ts>.yaml
```

Мониторинг по логам: строки `TYPED_METRIC_PROGRESS {...}` (`record_started`/`record_finished`
с `eg_type`, `raw_score`) и финальный `pass_complete`.

## Результаты и сравнение

- `typed_metrics.jsonl` — по строке на запись: `response_id`, `raw_score`, `components`
  (`eg_type`, `rp_grounded`, `rp_strict`, `cfi_type`, счётчики вершин/рёбер, `ungrounded_vertices`).
- `typed_metric_summary.json` — агрегат (`selected`, `ok`, `failed`, `empty_graph`, `alpha`).
- **Сравнение** с уже готовыми strict/support (их не перезапускаем): по тем же
  `response_id` считаем ROC-AUC для `CFI_type` и сопоставляем с существующими числами
  из [vertex-750qa-cache-replay-results.md](vertex-750qa-cache-replay-results.md).

## Известные риски / открытые пункты

1. **litellm/зависимости агента в контейнере.** Client-python (`/opt/hallu/client/bin/python`)
   собирался под cache-only replay (без LLM). Типизации нужны `litellm` и зависимости
   `dynamic_typing_agent`. Если их нет — первый лог покажет `ModuleNotFoundError`; тогда
   добавляем установку в джобу или в runtime-образ. (Проверяется первым прогоном.)
2. **Время на CPU.** Инстанс `c1.4` (CPU). Типизация 750 записей с HHEM NLI поштучно —
   долго; при необходимости поднять `--timeout-seconds` и/или GPU-инстанс.
3. **NLI включён** (HHEM). Тип берётся с verifier + best-effort фолбэком
   ([BEST_EFFORT_TYPING](../dynamic_typing_agent/docs/BEST_EFFORT_TYPING.md)).

## Ветка

Код на `zhenya` (коммит `d5b6808`). Целевая `codex/dynamic-entity-typing` (ba4ef76)
**разошлась** с `zhenya` (общий предок есть, но у каждой свои коммиты — простой перемоткой
не слить). Пока держим на `zhenya`; после успешного прогона пушим в общую ветку
(force-with-lease `zhenya:codex/dynamic-entity-typing` или через PR/merge).
