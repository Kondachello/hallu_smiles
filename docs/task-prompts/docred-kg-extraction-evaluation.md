# Prompt: оценка качества извлечения графов знаний на DocRED

Работай в существующем репозитории `/Users/maslovartemij/Projects/hallu_smiles`, в
текущей ветке `kggen-docred`. Не создавай ещё одну ветку, не удаляй кэши, архивы,
рендеренные DataSphere Jobs или результаты предыдущих RAGTruth-прогонов. Сначала
полностью прочитай `AGENTS.md`: в нём описаны обязательные научные и операционные
инварианты, завершённый статус `new-metrics`, текущий DocRED path, безопасная аутентификация DataSphere и
путь исполнения.

## Цель

Нужно независимо оценить **качество компонента извлечения графа знаний KGGen** на
DocRED. Это не новая метрика галлюцинаций и не повод менять формулы `strict`,
`support` или `support-critical` для RAGTruth.

Для каждого документа DocRED извлеки KG из текста тем же репозиторным путём KGGen,
сопоставь предсказанные направленные триплеты с размеченными триплетами DocRED и
ответь на два вопроса:

1. Какую долю размеченных ground-truth триплетов метод нашёл (recall)?
2. Какая доля извлечённых триплетов присутствует в разметке (precision)?

В отчёте второй показатель обязательно называй **gold-supported precision**, а не
абсолютной фактической точностью: DocRED размечает не все истинные факты из текста,
поэтому триплет, отсутствующий в его аннотации, не доказанно ложен.

## Сначала установи корректный экспериментальный протокол

1. Проверь используемый публичный релиз DocRED, его лицензию, состав файлов и
   наличие меток. Зафиксируй URL/версию/контрольную сумму или иной воспроизводимый
   идентификатор датасета. Не подменяй DocRED похожим датасетом.
2. Используй `train_annotated` только для разработки: выбора фиксированного
   extraction-конфига, глобальных порогов и политики сопоставления отношений. Не
   подсматривай gold-триплеты удерживаемого набора при выборе prompt, модели,
   алиасов, порога или маппинга предикатов.
3. Если у официального `test` нет публичных labels, проведи единственную итоговую
   оценку на публичном размеченном `dev` после заморозки дизайна по train. В тексте
   называй это *held-out development result*, а не невидимым benchmark test.
   Если доступны официальные размеченные test labels, используй их один раз после
   той же заморозки.
4. Используй уже согласованный детерминированный manifest: первые 50 документов
   `train_annotated` для calibration (первые 10 из них -- live smoke) и 200
   документов `dev` для held-out development evaluation, seed `42`. Всего 250
   уникальных документов. Не отбирай их по числу триплетов, сложности или
   результату экстрактора и не меняй manifest без нового указания пользователя.

## Что именно сопоставлять

DocRED задаёт сущности на уровне документа (`vertexSet`, несколько mention/alias на
одну entity) и направленные relation labels. Нельзя сравнивать триплеты просто по
строкам.

Построй явный, тестируемый и замороженный до held-out split слой alignment:

- Сначала нормализуй предсказанные subject/object и сопоставь их с document-level
  entity ID по title и всем mention aliases. Сохраняй направление head/tail.
  Неоднозначное сопоставление не угадывай: помечай его как ambiguous/unmatched и
  учитывай в диагностике.
- Сопоставляй relation KGGen с DocRED relation inventory через заранее описанный
  детерминированный evaluator. Используй human-readable descriptions из metadata
  DocRED и локальный/версионированный matcher; калибруй его только на train и затем
  заморозь. При tie или недостаточной уверенности выбирай `no match`.
- Не передавай KGGen полный список gold relation labels в основном extraction prompt:
  это превратит первичную оценку в DocRED-specific schema-constrained extractor.
  Такой вариант допустим лишь как отдельная явно помеченная ablation, а не как
  основной результат.
- После entity и relation alignment дедуплицируй предсказания по
  `(head_entity_id, relation_id, tail_entity_id)`. Один prediction может покрыть
  один gold triple; направление существенно. Не считай обратный триплет совпадением.
- Отдельно посчитай entity-pair coverage без relation label. Она диагностична и не
  заменяет полный triple metric.

Не используй внешний LLM как непрозрачного judge для основного matching, иначе
точность станет смесью качества экстрактора и ещё одной платной модели. Если
понадобится semantic relation matcher, он должен быть локальным/детерминированным,
иметь зафиксированную версию и отдельную offline-проверку на синтетических случаях.

## Обязательные метрики и честные знаменатели

На каждом итоговом split посчитай micro (по всем триплетам):

```text
matched = |predicted_aligned_triples ∩ gold_triples|
recall = matched / |gold_triples|
gold_supported_precision = matched / |predicted_aligned_triples|
F1 = harmonic_mean(recall, gold_supported_precision)
```

Также выведи явные числители/знаменатели и как минимум:

- число документов в manifest, extraction coverage и число ошибок;
- число gold triples, сырых предсказанных triples, entity-aligned predictions,
  relation-aligned predictions и дедуплицированных predictions;
- entity-pair precision/recall/F1 без relation label;
- типы неуспеха: extraction failure, malformed graph, unaligned entity, ambiguous
  entity, unmatched relation и дубликат;
- document-level non-parametric bootstrap 95% интервалы для triple recall,
  gold-supported precision и F1; при нулевом знаменателе зафиксируй правило заранее;
- при разумной стоимости — macro-by-relation диагностику и таблицу частых relation
  types, не маскируя long-tail micro-результатом.

Не исключай молча документы с ошибкой, пустым графом или нулём предсказаний. В
основном coverage-aware результате ошибка экстракции означает ноль предсказанных
триплетов для данного документа и уменьшает recall; параллельно можно привести
conditioned-on-success диагностику, но нельзя выдавать её за headline metric.

## Реализация и воспроизводимость

1. Осмотри существующие `src/extract.py`, тип `Graph`, кэши и DataSphere scripts.
   Добавь отдельный DocRED runner/evaluator и не ломай существующие RAGTruth пути.
   Используй отдельный cache namespace, включающий dataset release, document text,
   KGGen prompt/schema/runtime fingerprint и split/manifest provenance. Никогда не
   смешивай эти записи с RAGTruth cache lineage.
2. Кэш должен быть content-addressed, read-through и атомарным в соответствии с
   `AGENTS.md`. После основного прогона выполните cache-only replay: zero live API
   calls, те же metrics и те же cache inventories. Cache miss в replay — ошибка, а
   не разрешение вызвать Gemini.
3. До дорогостоящего запуска сделай локальные offline tests на небольших
   синтетических DocRED-подобных документах. Обязательно покрой: aliases, несколько
   mentions одной entity, direction, multi-label pair, дубликаты, ambiguity, пустой
   prediction, extraction failure, relation threshold/tie, micro denominator и
   невозможность использовать held-out labels в calibration.
4. Установи максимальный оценочный live-бюджет Gemini в EUR 10.5 из доступных
   EUR 13. Зафиксируй price snapshot в manifest; перед каждым холодным документом
   резервируй одну операцию, а после smoke не переходи к оставшимся 40/200,
   если консервативного резерва не хватает. При `budget_exhausted` сохрани
   checkpoint и заверши эксперимент как неполный -- не расширяй бюджет и не
   запускай ещё один платный Job без указания пользователя.
5. Сохраняй только безопасные diagnostics: не логируй API key, OAuth/IAM token,
   `Authorization`, prompt, completion, signed URL или cache key. Для DataSphere
   используй только неинтерактивный OAuth flow из `AGENTS.md`: секретный
   `YC_TOKEN`/`YC_OAUTH_TOKEN`, `YC_AUTH=OAUTH` и никакого profile fallback,
   `yc init` или browser login.
6. Не увеличивай concurrency ради скорости. Для живого Gemini прогона оставь
   консервативную сериализацию, bounded retry/backoff/pacing и сохранение каждого
   завершённого cache entry. Не превращай настоящую extraction/review ошибку в
   `unknown` или выдуманный пустой граф.

## Порядок работы

1. Кратко сообщи, какой DocRED release и публичный labelled split доступны, и
   предложи точную policy relation alignment до реализации.
2. Реализуй evaluator, manifests, provenance, cache-only replay и offline tests.
   Выполни `pytest -q`, syntax checks и проверку рендеринга DataSphere Job, если он
   нужен. Не делай платный полный прогон до успешного малого smoke run и понятной
   оценки масштаба/стоимости для зафиксированного manifest.
3. Для live запуска используй отдельные DocRED renderer/submitter/CPU template,
   immutable image, existing gateway gate и `scripts/submit_datasphere_vertex_docred_kg_eval.sh`.
   После submit создай монитор того же чата раз в 15 минут: проверяй status и
   ограниченный redacted tail, собирай live snapshots в новый `outputs/docred-...`
   каталог, различай outer/inner/retry progress. При terminal error сначала
   скачай archive/logs/diagnostics, сделай только доказанное compatible fix и
   не более одного serial cache-resume; jobs не должны пересекаться.
4. После согласованного живого прогона скачай архив Job в новый недеструктивный
   `outputs/docred-...` каталог; финальные числа бери только из архива, а не из
   консольного лога.
5. Подготовь отдельный `docs/docred-kg-extraction-results.tex` с методом
   alignment, версиями данных/модели, manifest, coverage, всеми метриками и
   интервалами, caveat о неполной аннотации DocRED и limitations. Не переписывай
   `docs/support-critical-100qa-results.tex` или
   `docs/support-critical-750qa-results.tex`. Собери/проверь TeX, если доступен
   подходящий движок.

## Финальная сдача

В финальном сообщении дай ссылки на изменённые файлы, archive/output directory и
результирующий TeX; перечисли точный manifest, live/cache-only usage, coverage,
micro triple recall, gold-supported precision, F1 и интервалы. Явно отдели
подтверждённые числа от ограничений. Не заявляй, что отсутствующие в DocRED
предсказания являются ложными фактами.
