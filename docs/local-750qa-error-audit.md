# Локальный аудит ошибок HalluGraph на 750-QA

Этот набор даёт другому разработчику воспроизводимый путь от скачанного
запечатанного архива к HTML-предпросмотру и отдельным пакетам для агента-аудитора.
Все команды являются постфактум-анализом: они не запускают DataSphere, KGGen,
HalluGraph, GraphEval, LLM или gateway и не изменяют запечатанный архив.

## Что нужно получить заранее

В репозиторий не включены большой архив 750-QA и официальный набор RAGTruth.
Нужны три локальных пути:

```text
<archive-dir>/                         # extracted historical-cache-replay
<responses>/response.jsonl             # официальный RAGTruth response.jsonl
<analysis-dir>/                         # новая внешняя папка для HTML и пакетов
```

Для исторического 750-QA запуска пути обычно выглядят так:

```text
outputs/datasphere-results/zhenya-750all-final/
├── source/response.jsonl
└── zhenya-750all-20260726-165854/historical-cache-replay/
```

`response.jsonl` берётся из официального
[репозитория RAGTruth](https://github.com/ParticleMedia/RAGTruth/blob/main/dataset/response.jsonl).
До анализа убедитесь, что архив распакован полностью, включая
`instances.no_gold.jsonl`, `predictions/raw_predictions.jsonl`,
`shared_graphs/graph_index.jsonl` и `prediction_seal.json`.

## 1. Получить код

```powershell
git clone <URL_РЕПОЗИТОРИЯ> hallu_smiles
Set-Location hallu_smiles
git switch codex/experiment-framework-spec
git pull --ff-only
```

Скрипты используют только стандартную библиотеку Python и код репозитория.
Запускайте команды из корня `hallu_smiles` тем же Python, которым запускаются
тесты проекта.

## 2. Построить HTML и метрики

Сначала создайте результат постфактум-оценки. Пороги выбираются только на
обучающей части по максимальному F1, после чего фиксируются для тестовой части.

```powershell
python scripts/build_historical_replay_gold_audit.py `
  --archive-dir <archive-dir> `
  --responses <responses>/response.jsonl `
  --output-html <analysis-dir>/gold-audit.html `
  --output-metrics <analysis-dir>/gold-audit-metrics.json
```

Команда проверит sealing исходного архива, затем присоединит официальные метки
только для анализа. `gold-audit-metrics.json` обязателен для следующего шага:
в нём зафиксированы происхождение и значения порогов.

## 3. Открыть HTML и вручную посмотреть пример

Откройте `<analysis-dir>/gold-audit.html` двойным щелчком либо в PowerShell:

```powershell
Start-Process <analysis-dir>/gold-audit.html
```

В странице:

1. В фильтре «Показать» выберите «ошибка HalluGraph».
2. При необходимости ограничьте `train` или `test`, найдите `response_id` либо
   отсортируйте по баллу или расхождению методов.
3. Щёлкните строку. Слева появятся Query, Context и Response; справа — решения,
   баллы, компоненты HalluGraph, тройки GraphEval и вкладки графов контекста,
   запроса и ответа.
4. Учитывайте предупреждение наверху: в историческом Job `empty_graph` был
   ошибочно учтён как общая ошибка завершения. В HTML такие строки честно
   помечены «без балла» и не входят в численные метрики.

Это удобный ручной просмотр до или после передачи случая агенту.

## 4. Сформировать пакеты для агентов-аудиторов

Экспорт всех численно оцениваемых ложных срабатываний HalluGraph:

```powershell
python scripts/export_historical_replay_audit_case.py `
  --archive-dir <archive-dir> `
  --responses <responses>/response.jsonl `
  --metrics <analysis-dir>/gold-audit-metrics.json `
  --hallugraph-errors fp `
  --output-dir <analysis-dir>/audit-packages/fp
```

Для пропусков галлюцинаций замените `fp` на `fn`; для обоих классов — на `all`.
Команда создаёт:

```text
audit-packages/fp/
├── audit-manifest.jsonl
├── audit-case-<response_id>.json
└── audit-case-<response_id>.md
```

`audit-manifest.jsonl` — компактный список для распределения задач. Полный JSON
включает тексты, три графа, исходную RAGTruth-разметку, необрезанные артефакты
обоих методов, баллы, пороги, решения, хэши и проверку sealing.

Для одного конкретного случая:

```powershell
python scripts/export_historical_replay_audit_case.py `
  --archive-dir <archive-dir> `
  --responses <responses>/response.jsonl `
  --metrics <analysis-dir>/gold-audit-metrics.json `
  --response-id 12415 `
  --output-dir <analysis-dir>/audit-packages/12415
```

По умолчанию команда не перезаписывает уже подготовленные пакеты. Для осознанной
пересборки добавьте `--overwrite` либо укажите новую папку.

## 5. Передать кейс агенту

Системный промпт находится в
[`docs/hallugraph-error-audit-system-prompt.md`](hallugraph-error-audit-system-prompt.md).
Передайте его как системную инструкцию, а содержимое одного
`audit-case-<response_id>.json` — как пользовательский ввод или прикреплённый
файл. Короткий одноимённый Markdown-файл рядом с JSON содержит только сводку и
правило применения пакета.

Промпт требует сначала провести независимый аудит HalluGraph, отделить эффект
KGGen с помощью контрфактических проверок, затем вторично разобрать GraphEval.
В конце он заставляет агента вернуть стабильно размеченные теги и JSON-резюме,
что позволяет затем кластеризовать сотни случаев.

## Проверка изменений

```powershell
python -m pytest -c pytest.framework.ini --rootdir=. -p no:cacheprovider tests/scripts/test_export_historical_replay_audit_case.py
python -m py_compile scripts/build_historical_replay_gold_audit.py scripts/export_historical_replay_audit_case.py
```

Ни одна из этих команд не требует сетевого доступа и не вызывает детекторы.
