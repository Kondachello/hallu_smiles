# 04. Результаты прогона

Прогон: `runs/typing-demo/` (20 примеров, gateway LLM + локальный HHEM, конфиг `live-gateway-hhem.yaml`).

## Сводка

- **19/20 успешно**, 1 упал.
- `dt-014-event-location-time` — **не логическая** ошибка, а сеть:
  `TransportError('entity_type_decision: live completion failed after 5/5 attempt(s): Timeout')`.
  Транзиентный таймаут gateway; кейс можно догнать отдельно (остальные 19 из кэша).
- Всего типов в реестрах: **106**. Из них меток `entity`: **19** — это по одному **корню иерархии**
  на каждый успешный реестр (19 реестров), а **не** результат отката. Корень всегда присутствует в
  списке типов и не является «плохим» назначением.
- Назначений, схлопнутых в **голый** `entity` (единственный тип у реальной сущности): **2** — и только
  там, где модель не предложила ни одной гипотезы (корректный остаточный случай).
- **Best-effort фолбэк** (новое поведение вместо `entity`, `evidence_level=unknown`): **4 типа**.

## Best-effort в действии

Сработал ровно там, где раньше был бы бесполезный `entity`:

| Кейс | Best-effort тип | Комментарий |
|------|-----------------|-------------|
| `dt-020-unsupported-specialization` | **`endeavor`** | Кейс специально про «NLI не подтвердит специализацию» — вместо `entity` сохранён информативный тип |
| `dt-007-python-cpython-version` | `software` | |
| `dt-015-document-vs-copy` | `archival repository` | |
| `dt-016-drug-dose-quantity` | `medical professional` | |

Каждый такой выбор помечен `evidence_level=unknown` и сопровождается событием
`type_best_effort_fallback` в `execution_trace.json` с полным ранжированным списком кандидатов.

## Типы по кейсам

Осмысленные, специфичные типы — то, ради чего делалось изменение (`entity` в конце каждого списка —
корень иерархии):

| Кейс | Типы |
|------|------|
| dt-001-bank-generalization | organization, debt instrument, financial institution, commercial bank, company |
| dt-002-legal-contextual-role | role, legal claim, organization, company |
| dt-003-brca1-gene-protein | biological process, gene, protein |
| dt-004-insulin-multi-aspect | medical intervention, chemical compound, person, injection, hormone, sugar, biological substance |
| dt-005-virus-disease-direction | biological entity, virus, infectious disease, disease |
| dt-006-apple-metonymy | brand, stock, percentage, product, organization, event, financial instrument, company |
| dt-007-python-cpython-version | API, software version, **software***, programming language, reference implementation, implementation |
| dt-008-log4shell-cve-library | CVE ID, vulnerability, software library |
| dt-009-webb-nircam-part-whole | space observatory, infrared scientific instrument, observatory, instrument |
| dt-010-aircraft-model-instance | individual aircraft, city, aircraft model, aircraft |
| dt-011-university-multiple-parents | undergraduate student, research institute, institution, university, educational institution, public legal entity |
| dt-012-person-temporary-role | person, clinical trial, cardiologist, professional role |
| dt-013-mercury-homonym | marine engine, celestial body, organization, planet, engine, company |
| dt-014-event-location-time | — (source failed: gateway Timeout) |
| dt-015-document-vs-copy | paper copy, document, treaty, year, **archival repository*** |
| dt-016-drug-dose-quantity | administration frequency, treatment duration, drug dosage quantity, medication, **medical professional*** |
| dt-017-company-vs-service | managed database service, organization, database service, data record |
| dt-018-planet-vs-spacecraft | spacecraft, planet |
| dt-019-query-only-entity | device, document, university, organization |
| dt-020-unsupported-specialization | **endeavor***, bank, financial institution |

`*` — best-effort тип (`evidence_level=unknown`).

## Как посмотреть

- HTML-viewer: `runs/typing-demo/viewer/index.html` (общая страница + страница на каждый кейс).
- Сырые артефакты кейса: `runs/typing-demo/<case>/`
  - `source_registry.json` — реестр типов (`registry.types`, `registry.assignments`, `registry.nli_results`);
  - `execution_trace.json` — трейс узлов, включая `source_cache` (route) и `type_best_effort_fallback`;
  - `input_snapshot.json`, `manifest.json`.

## Как перепроверить локально (без повторного прогона)

```bash
cd dynamic_typing_agent
python - <<'PY'
import json, glob, os
for f in sorted(glob.glob("runs/typing-demo/*/source_registry.json")):
    reg = json.load(open(f, encoding="utf-8")).get("registry")
    name = os.path.basename(os.path.dirname(f))
    if reg is None:
        print(name, "(source failed)"); continue
    labels = [t["label"] for t in reg["types"]]
    unknown = [t["label"] for t in reg["types"] if t.get("evidence_level") == "unknown"]
    print(name, labels, "best-effort:", unknown)
PY
```
