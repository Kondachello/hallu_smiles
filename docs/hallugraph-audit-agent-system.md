# Агентная система аудита ошибок HalluGraph

Надстройка над локальным аудитом (`docs/local-750qa-error-audit.md`): экспортёр
даёт пакеты кейсов, эта система прогоняет по одному агенту-аудитору на кейс и
собирает два итоговых файла.

Как и весь аудит, система работает **только постфактум**: не запускает DataSphere,
KGGen, HalluGraph, GraphEval, LLM или gateway и не трогает запечатанный архив.

## Из чего состоит

| Компонент | Файл | Роль |
|---|---|---|
| Базовый чек-лист | `docs/hallugraph-error-audit-system-prompt.md` | системный промпт воркера; **фиксирован** на весь прогон |
| Проекции кейсов | `scripts/audit_agents/prepare_case_projections.py` | режет пакет на `case-<id>.hg.json` / `.ge.json` |
| Реестр аспектов | `scripts/audit_agents/aspect_registry.py` | append-only реестр + лог происхождения |
| Оркестратор | `scripts/audit_agents/hallugraph-case-audit.workflow.js` | волны воркеров + регистратор |
| Сборка FILE-1 | `scripts/audit_agents/assemble_audit_report.py` | покейсовый отчёт + индекс |

## Два инварианта, на которых всё держится

**1. Метод под аудитом видит только себя.** Промпт просит «не считай GraphEval
эталоном», но пакет всё равно показывал его вердикт. `prepare_case_projections.py`
удаляет чужой метод физически и проверяет результат регуляркой: если в проекции
осталось хоть одно упоминание скрытого метода — команда падает с `LeakError`.
Скрытый метод даже не называется: знать, какой детектор от тебя спрятали, — тоже
якорь. Сравнение методов делает отдельная поздняя стадия.

**2. Реестр аспектов пишет ровно один агент за волну.** Воркеры ничего не пишут
в общие файлы — они возвращают предложенный аспект в structured output. Между
волнами один регистратор дедуплицирует предложения и добавляет принятые. Отсюда
и волны: внутри волны реестр заморожен, поэтому все агенты волны видят одно и то
же состояние, а append-only файлы не рвутся конкурентной записью.

Регистратор намеренно строгий. Реестр читает каждый следующий агент; раздутый
реестр съедает внимание и провоцирует формальные отписки. Планка — аспект должен
менять то, *как* агент смотрит на кейс, а не просто называть найденное.

## Прогон

```bash
PY=.venv/bin/python
ANALYSIS=<analysis-dir>

# 0. Пакеты кейсов (см. docs/local-750qa-error-audit.md, шаги 2-5)
$PY scripts/export_historical_replay_audit_case.py \
  --archive-dir <archive-dir> --responses <responses>/response.jsonl \
  --metrics $ANALYSIS/gold-audit-metrics.json \
  --hallugraph-errors all --output-dir $ANALYSIS/audit-packages/all

# 1. Проекции: HalluGraph-версия без чужого вердикта
$PY scripts/audit_agents/prepare_case_projections.py \
  --input-dir $ANALYSIS/audit-packages/all \
  --output-dir $ANALYSIS/projections --method hallugraph

# 2. Прогон агентов (волнами) — через Workflow с
#    scripts/audit_agents/hallugraph-case-audit.workflow.js
#    args: caseIds, projectionDir, outputDir, promptPath, registry*, python, waveSize

# 3. FILE-1
$PY scripts/audit_agents/assemble_audit_report.py \
  --audits-dir $ANALYSIS/audits \
  --manifest $ANALYSIS/audit-packages/all/audit-manifest.jsonl \
  --summaries $ANALYSIS/audits/summaries.jsonl \
  --output $ANALYSIS/FILE1-hallugraph-case-audits.md
```

FILE-2 (кластеризация всех кейсов в таксономию ошибок) строится отдельной
стадией поверх FILE-1.

## Что получается на выходе

```
<analysis-dir>/
  projections/case-<id>.hg.json          вход воркера
  audits/<id>.hg.md                      покейсовый отчёт агента
  audits/summaries.jsonl                 структурированные возвраты
  aspects/aspects.jsonl                  принятые аспекты (append-only)
  aspects/aspects.md                     рендер для промпта воркеров
  aspects/aspects-log.jsonl              все предложения: принятые и отклонённые
  FILE1-hallugraph-case-audits.md        сухой покейсовый разбор
  FILE2-hallugraph-error-taxonomy.md     кластеры и аудит метода
```

`aspects-log.jsonl` отвечает на вопрос «откуда взялся этот аспект»: в каждой
строке кейс, волна, агент, статус и причина решения регистратора.

## Порядок стадий

1. **HalluGraph** — воркеры видят только HalluGraph.
2. **GraphEval** — то же самое, свой реестр аспектов, независимо.
3. **Сравнение** — агент получает оба готовых аудита одного кейса.
4. **Агрегация** — FILE-1 → кластеризация → FILE-2.

Стадия 3 не должна менять уже зафиксированные первопричины стадий 1-2, если не
нашлось прямого нового доказательства.
