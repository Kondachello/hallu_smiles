# Результаты 750-QA cache-only replay

Полный прогон HalluGraph и GraphEval по историческому 750-QA кэшу в режиме
`cache_only` (без единого вызова LLM). Ниже — итоговая сводка и способ её
воспроизвести из сохранённого архива.

## Идентичность прогона

| Параметр | Значение |
|---|---|
| DataSphere Job | `bt1583ah6e0o0jm9lqd3` |
| Commit | `4266dc6` (ветка `zhenya`) |
| Runtime protocol | `hallu-datasphere-cpu-vertex-v1` |
| LLM runtime fingerprint | `vertex-gateway:9ba169c4f2de8a246c756948b24a3860a54cc419957004d9ed351c2ad538b3bd` |
| Gateway manifest sha256 | `9407591410b215ba41478290526acd3a4ea32f3dd70a63076c6394c95e37c845` |
| QA-выборка | `qa_sample_size=750`, `qa_test_fraction=0.2`, `sample_seed=42` |
| Режим выбора записей | `replay_count=all` (все полностью закэшированные записи) |
| Cache key schema (попадания) | `kggen-v11-pre-length-retry` (legacy) — 2247/2250 |

## Источники графов (read-through цепочка)

Кэш собирается из ДВУХ каталогов на Project storage (порядок по приоритету):

1. `checkpoints/vertex-qa/qa-750-test-150-cv-5/baseline-v1-<manifest>/kg`
   — основной writable-кэш 750-прогона (≈1944 файла на момент запуска);
2. `checkpoints/vertex-qa/qa-100-test-20-cv-5/<commit>-<manifest>/kg`
   — исторический 100-QA кэш, из которого read-through читаются графы для
   100 записей, общих у 750- и 100-выборок (300 файлов).

`graph_sources` в отчёте: `["historical_100qa", "historical_lineage_0"]` —
доказательство, что задействованы оба источника.

## Итоговые метрики

| Показатель | Значение |
|---|---|
| Записей отреплеено | **749 / 750** |
| Предсказаний | 1498 (749 × 2 детектора) |
| Использовано кэш-графов | 2228 |
| `kggen_api_calls` | **0** |
| `gateway_llm_calls` | **0** |
| `grapheval_extractor_calls` | **0** |
| `detector_statuses` | `hallugraph: ok`, `grapheval: ok` |
| `validation.valid` | `true` |

Разбивка по статусам (на каждый детектор): **729 `ok` + 20 `empty_graph`**.
`empty_graph` — легитимный, не-LLM исход: пустой/полностью невалидный граф
ответа даёт `raw_score=None` (см. `graph_eval/DEVIATIONS.md` §9). Это НЕ ошибка.

Распределение скоров (по 729 записям с числовым скором):

| Детектор | n | mean | median | min | max |
|---|---|---|---|---|---|
| HalluGraph | 729 | 0.427 | 0.408 | 0.000 | 0.926 |
| GraphEval | 729 | 0.600 | 0.726 | 0.005 | 0.998 |

## Единственная незакэшированная запись

`response_id=17712`, `source_id=12448` — все три роли отсутствовали в кэше на
момент прогона (их дописывает поздний extraction-run `r2`). Поэтому
`complete_records=749`, а не 750. Когда `r2` завершится и допишет эти 3 графа,
тот же прогон даст ровно 750/750 без изменений кода.

## Как воспроизвести сводку из архива (без повторного прогона)

Архив: `historical-qa-cache-replay-zhenya-750all-20260726-165854.tar.gz`.
Все результаты записаны на диск ДО финальной проверки, поэтому пересчёт итога не
требует повторного запуска Job:

```bash
tar -xzf historical-qa-cache-replay-zhenya-750all-*.tar.gz -C /tmp/x
REP=$(find /tmp/x -name historical_qa_cache_replay_report.json)
RAW=$(find /tmp/x -name raw_predictions.jsonl)
python - "$REP" "$RAW" <<'PY'
import json,sys,statistics
from collections import Counter
rep=json.load(open(sys.argv[1],encoding="utf-8"))
rows=[json.loads(l) for l in open(sys.argv[2],encoding="utf-8")]
print("replay_count           :", rep["replay_count"])
print("kggen_api_calls        :", rep["kggen_api_calls"])
print("grapheval_extractor_calls:", rep["grapheval_extractor_calls"])
print("graph_sources          :", rep["graph_sources"])
print("by method/status       :", dict(Counter((r["method"],r["status"]) for r in rows)))
for m in ("hallugraph","grapheval"):
    sc=[float(r["raw_score"]) for r in rows if r["method"]==m and r.get("raw_score") is not None]
    print(f"{m}: n={len(sc)} mean={statistics.mean(sc):.4f} median={statistics.median(sc):.4f}")
PY
```

## Историческая заметка: почему это было так сложно

До коммитов `4266dc6`/`566f5a8` прогон падал с `available=0` или ошибкой в конце.
Три отдельных бага (подробности и инструкция по запуску —
`docs/datasphere-historical-qa-cache-replay.md`):

1. в replay не пробрасывался `--qa-sample-size`, поэтому вместо 750-выборки
   материализовалась выборка из 100 записей — другой набор текстов, не
   совпадающий с 750-кэшем → все ключи мимо;
2. подключался только `baseline/kg`; read-through из 100-QA кэша не был
   настроен, из-за чего 100 общих записей промахивались;
3. финальный инвариант считал `empty_graph` провалом, из-за чего успешный
   прогон падал в самом конце.
