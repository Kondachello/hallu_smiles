# Best-effort типизация (сводка)

> Одностраничная сводка. Подробности — в папке [`best-effort-typing/`](best-effort-typing/README.md).

## Что было сделано

Устранена проблема, из-за которой сущности массово получали бесполезный служебный тип `entity`:
если NLI (HHEM) не подтверждал ни одной гипотезы, типизация откатывалась к структурному корню
`T-ENTITY`. Алгоритм обновлён `entity-by-entity-nli-v3` → **`v4-best-effort`**.

**Суть нового поведения:**
1. Цикл до `max_entity_attempts` гипотез (дефолт **2 → 3**, настраивается).
2. При «слабом» NLI (только neutral) — переспрос новой гипотезы (`retry_on_neutral`, дефолт `true`).
3. После исчерпания попыток — берётся **самый правдоподобный по NLI** кандидат
   (`entailed > neutral > contradicted`, затем по силе доказательства), а не корень.
   `entity` остаётся **только если модель не предложила вообще ничего**.
4. Best-effort выбор помечается `evidence_level=unknown`, получает `reason` и событие
   `type_best_effort_fallback` — низкая уверенность видна и аудируется.

Коммит: `ba665cf`. Затронуто: `quality_workflow.py`, `agent.py`, `config/live-gateway-hhem.yaml`,
`tests/test_quality_invariants.py`.

## Проверка

- Оффлайн: **57 тестов зелёные**.
- Live-демо (gateway LLM + локальный HHEM) на 20 примерах: **19/20 успешно**
  (1 — транзиентный таймаут gateway, не логика). Best-effort сработал в 4 местах вместо `entity`
  (`endeavor`, `software`, `archival repository`, `medical professional`); голым `entity`
  схлопнулись лишь 2 назначения (случаи «нет ни одной гипотезы»).
- Артефакты: `runs/typing-demo/`, HTML: `runs/typing-demo/viewer/index.html`.

## Детальные документы

| Файл | О чём |
|------|-------|
| [best-effort-typing/README.md](best-effort-typing/README.md) | Обзор + оглавление |
| [best-effort-typing/01-problem-and-solution.md](best-effort-typing/01-problem-and-solution.md) | Проблема и алгоритм по шагам |
| [best-effort-typing/02-code-changes.md](best-effort-typing/02-code-changes.md) | Точные изменения в коде и конфиге |
| [best-effort-typing/03-live-run-setup.md](best-effort-typing/03-live-run-setup.md) | Live-окружение, блокеры, команда воспроизведения |
| [best-effort-typing/04-results.md](best-effort-typing/04-results.md) | Итоги прогона, типы по кейсам |
