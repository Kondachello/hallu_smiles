# 02. Изменения в коде

Коммит `ba665cf` — «Typing: best-effort fallback instead of collapsing to the root 'entity'».

```
 src/hallugraph_dynamic_typing/agent.py             |   8 +-
 src/hallugraph_dynamic_typing/quality_workflow.py  | 111 +++++++++++++++++-
 tests/test_quality_invariants.py                   |   7 +-
 config/live-gateway-hhem.yaml                      | (обновлён отдельно под демо)
```

## 1. `quality_workflow.py` — ядро изменения

### Версия и таблицы ранжирования (верх модуля)

```python
ROOT_TYPE_ID = "T-ENTITY"
ALGORITHM_VERSION = "entity-by-entity-nli-v4-best-effort"   # было "…-v3"

_VERDICT_RANK = {"entailed": 3, "neutral": 2, "contradicted": 1}
_EVIDENCE_RANK = {
    EvidenceLevel.SOURCE_ENTAILED: 3,
    EvidenceLevel.DEFINITION_ONLY: 2,
    EvidenceLevel.UNKNOWN: 1,
}
```

### Конструктор `QualityTypingWorkflow.__init__`

- `max_entity_attempts: int = 2` → **`= 3`**;
- добавлен параметр `retry_on_neutral: bool = True` и поле `self.retry_on_neutral`.

### `type_source(...)` — накопление кандидатов

Перед циклом попыток заводится пул всех проверенных семантических кандидатов:

```python
best_effort_pool: list[dict[str, Any]] = []
```

Внутри цикла NLI-проверки, для каждого не-корневого типа кандидат кладётся в пул **до** решения о приёме:

```python
if type_id != ROOT_TYPE_ID:
    candidate_def = proposed[target_ref][1] if target_ref in proposed else types.get(type_id)
    best_effort_pool.append({
        "type_id": type_id, "label": label, "verdict": result.verdict,
        "evidence_level": result.evidence_level, "definition": candidate_def, "attempt": attempt,
    })
```

### Новые условия остановки цикла

Вместо прежнего единственного условия
`if semantic_accepted or not semantic_targets or attempt == max: break`
теперь явная логика по вердиктам текущей попытки:

```python
this_attempt   = [i for i in best_effort_pool if i["attempt"] == attempt]
attempt_entailed = [i for i in this_attempt if i["verdict"] == "entailed"]
attempt_neutral  = [i for i in this_attempt if i["verdict"] == "neutral"]

if attempt_entailed:                                   break   # нашли entailed — стоп
if not semantic_targets:                               break   # типизировать нечего
if attempt_neutral and not self.retry_on_neutral:      break   # neutral принимаем, если не переспрашиваем
if attempt == self.max_entity_attempts:                break   # исчерпали бюджет
# иначе — переспрос с уточнённой инструкцией (см. previous_attempt.instruction)
```

### Финальный best-effort фолбэк

После цикла, если нет ни одного подтверждённого семантического типа, но пул непуст:

```python
semantic_ids = tuple(sorted({tid for tid in accepted_ids if tid != ROOT_TYPE_ID}))
if not semantic_ids and best_effort_pool:
    ranked = sorted(
        best_effort_pool,
        key=lambda i: (_VERDICT_RANK.get(i["verdict"], 0), _EVIDENCE_RANK.get(i["evidence_level"], 0)),
        reverse=True,
    )
    best = ranked[0]
    chosen_id = str(best["type_id"])
    if best["definition"] is not None:
        types.setdefault(chosen_id, best["definition"].model_copy(
            update={"evidence_level": EvidenceLevel.UNKNOWN}))
    if chosen_id in types:
        semantic_ids = (chosen_id,)
        last_reason = (f"Best-effort type '{best['label']}' kept after "
                       f"{self.max_entity_attempts} attempt(s); highest NLI verdict was "
                       f"'{best['verdict']}' with no source-entailed type. "
                       f"Chosen over the structural root 'entity'.")
        events.append({"event": "node_completed", "node": "type_best_effort_fallback",
                       "inputs": {...}, "outputs": {"chosen_type_id": chosen_id,
                       "chosen_label": best["label"], "chosen_verdict": best["verdict"],
                       "ranked_candidates": [ ... весь ranked ... ]}})

final_ids = semantic_ids or (ROOT_TYPE_ID,)   # корень — только если пул был пуст
```

## 2. `agent.py` — проброс параметров

- `DynamicTypingAgent.__init__`: `max_entity_attempts` дефолт `2 → 3`, добавлен `retry_on_neutral: bool = True` и поле.
- `from_config(...)`: читает `retry_on_neutral` из `source`-секции конфига (`bool(source_config.get("retry_on_neutral", True))`),
  дефолт `max_entity_attempts` тоже `2 → 3`.
- Место сборки `QualityTypingWorkflow(...)`: добавлен `retry_on_neutral=self.retry_on_neutral`.

## 3. `config/live-gateway-hhem.yaml` — под демо

Секция `source` приведена к новой политике:

```yaml
source:
  entity_batch_size: 1
  max_entity_attempts: 3      # было 2
  retry_on_neutral: true
```

## 4. `tests/test_quality_invariants.py` — обновление под новую политику

Тест `test_source_neutral_can_finalize_but_contradiction_falls_back` кодировал **старое** поведение
(«contradicted → откат к `entity`»). Переименован в
`test_source_neutral_finalizes_and_contradiction_keeps_best_effort_type` и параметризация обновлена:

```python
# было:  ("contradicted", "entity",       "source_entailed")
# стало: ("contradicted", "organization", "unknown")
```

То есть единственный contradicted-кандидат теперь сохраняется как best-effort
(`organization`, `evidence_level=unknown`), а не схлопывается в `entity`.

**Проверка:** весь оффлайн-набор (57 тестов) зелёный.
