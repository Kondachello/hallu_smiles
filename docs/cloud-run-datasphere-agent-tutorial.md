# Туториал для агентов: общий Cloud Run LLM gateway + DataSphere

Этот документ описывает **инфраструктурный** путь для новой задачи, в которой
нужно оставить существующий baseline и заменить только транспорт к LLM:

```text
DataSphere CPU Job ──HTTPS + Bearer secret──> Cloud Run gateway ──ADC──> Vertex AI
       │                                             │
       ├─ baseline / data / metrics / caches          └─ Gemini model
       └─ artifacts, logs без prompt/secret
```

Здесь нет требований к HalluGraph, конкретной метрике или датасету. Их должен
задавать владелец нового эксперимента. Gateway — только безопасный
OpenAI-совместимый адаптер к Vertex AI.

## 1. Что уже предоставлено команде

| Ресурс | Как использовать |
|---|---|
| Cloud Run gateway | `https://hallu-vertex-gateway-453887629111.europe-west4.run.app` |
| Модель | логическое имя `openai/gemini-2.5-flash` |
| DataSphere secret | `HALLU_GATEWAY_API_KEY`; Project автоматически передаёт его в окружение Job |
| Идентичность Vertex | только service account самого Cloud Run через ADC |

`HALLU_GATEWAY_API_KEY` — общий секрет, но не общий текстовый параметр.
Агент **никогда** не запрашивает, не выводит, не коммитит и не кладёт его в
YAML. Внутри DataSphere он просто уже доступен как environment variable.

Не нужны и запрещены: Google service-account JSON, `GOOGLE_APPLICATION_CREDENTIALS`,
Gemini Developer API key и прямой вызов Vertex из DataSphere.

## 2. Контракт gateway

Каждый запрос требует:

```http
Authorization: Bearer $HALLU_GATEWAY_API_KEY
```

Доступны защищённые endpoint'ы:

| Endpoint | Назначение |
|---|---|
| `GET /healthz` | минимальная проверка доступности |
| `GET /v1/hallu/manifest` | неизменяемая идентичность protocol/model/revision; обязательна перед реальным запуском |
| `POST /v1/chat/completions` | ограниченное OpenAI Chat Completions API |

Поддерживаются `model`, текстовые `messages` с ролями `system`/`user`/`assistant`,
`temperature`, `max_tokens` и строгий
`response_format={"type":"json_schema", ...}`. Ответ содержит один choice,
finish reason, usage и fingerprint модели. Gateway намеренно не поддерживает
streaming, tools, multipart content, `n > 1` и произвольную модель.

Пример минимального клиентского слоя (не логируйте ни API key, ни prompt):

```python
import os
from openai import OpenAI

gateway_url = os.environ["HALLU_GATEWAY_URL"].rstrip("/")
client = OpenAI(
    base_url=f"{gateway_url}/v1",
    api_key=os.environ["HALLU_GATEWAY_API_KEY"],
)

response = client.chat.completions.create(
    model="openai/gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "Return only the requested JSON."},
        {"role": "user", "content": "..."},
    ],
    temperature=0,
    max_tokens=1024,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": {"type": "object"}},
    },
)
```

Gateway проверяет формат ответа, но **ваш pipeline обязан локально валидировать
полученный JSON Schema**. Это защищает от усечённого или семантически неверного
ответа модели.

## 3. Как подключить новую задачу, не изменяя baseline

Работайте в следующем порядке.

1. Сначала зафиксируйте baseline: commit, данные, deterministic manifest/split,
   исходные метрики и cache layout. Не меняйте их одновременно с transport.
2. Введите один тонкий LLM adapter. Единственные его входы: `api_base`,
   `api_key_env`, logical model, timeout/retry policy и structured-output schema.
   Остальная научная логика не должна знать, что используется Vertex или Cloud Run.
3. В конфиге установите только:

   ```yaml
   llm:
     model: "openai/gemini-2.5-flash"
     api_base: "https://hallu-vertex-gateway-453887629111.europe-west4.run.app/v1"
     api_key_env: "HALLU_GATEWAY_API_KEY"
     temperature: 0.0
   ```

   Не помещайте значение ключа в конфиг. Если runtime config создаётся внутри
   Job, сохраняйте в artifact только его redacted-вариант.
4. Сначала запустите маленький deterministic probe: 2–3 реальных записи,
   обязательный structured-output path и один cache-only replay. Это проверка
   совместимости, а не оценка качества модели.
5. Только затем запускайте полный эксперимент. Baseline и новый вариант должны
   использовать один manifest, split, preprocessing и заранее определённый
   analysis plan.

## 4. Обязательная модель кэша

Кэш — не просто ускорение: он защищает воспроизводимость. Ключ каждого
результата LLM должен включать как минимум:

```text
version of prompt/schema + logical model + gateway manifest SHA-256
+ generation parameters + canonical input text (+ retrieved evidence, if any)
```

Перед вычислением job запрашивает `/v1/hallu/manifest`, канонически
хеширует JSON и сохраняет manifest рядом с результатами. При смене Cloud Run
revision, модели, release или protocol hash меняется — старый ответ не должен
молча использоваться.

Пишите cache entries атомарно: во временный уникальный файл, затем `os.replace`.
Это обязательно при concurrency и при повторном запуске после прерывания.

Для нового метода рядом с baseline используйте **новый namespace** кэша,
например `cache/<baseline-id>/new-method-v1/`; baseline cache не перезаписывайте.
Сначала делайте read-through прежнего кэша там, где формат и fingerprint
совместимы. В режиме `--cache-only` отсутствие записи должно быть ошибкой,
а не причиной скрытого сетевого вызова.

Критерий готовности: второй идентичный запуск делает `0` HTTP calls и выдаёт
те же hash результатов.

## 5. Retry и обработка ответов модели

У агента должны быть две разные ветки ошибок.

| Событие | Действие |
|---|---|
| 429, 500–599, timeout, DNS/connection/HTTP2 reset | повторять с exponential backoff + jitter до восстановления либо общего wall-time Job; не помечать пример ошибкой из-за временной сети |
| 400, 401, 403, 404 | fail fast: это конфигурация, схема, secret или доступ; повтор не поможет |
| JSON не парсится, schema не проходит, модель вернула `finish_reason=length` | ограниченное число corrective re-prompt; затем явно сохранить controlled fallback или failed item по заранее принятому протоколу |
| Ошибка в одной записи | сохранить диагностический artifact с причиной и id; policy продолжения/остановки должна быть определена до запуска |

Не путайте устойчивость с подгонкой эксперимента: retry допустим только для
транспортной ошибки. Менять prompt, ответ, label или порог после того, как
увидели конкретный пример, нельзя.

## 6. Шаблон DataSphere Job

В YAML не должно быть ключа. Достаточно передать URL и обычные параметры;
секрет DataSphere вставит сам.

```yaml
cmd: |
  set -euo pipefail
  export HALLU_GATEWAY_URL="https://hallu-vertex-gateway-453887629111.europe-west4.run.app"
  export RUN_ROOT="$DS_PROJECT_HOME/runs/$RUN_ID"
  mkdir -p "$RUN_ROOT"

  # 1. authenticated manifest -> runtime config (redacted в artifact)
  # 2. deterministic small preflight
  # 3. main baseline/new-method run
  # 4. cache-only replay
  # 5. archive artifacts
  timeout --signal=TERM --kill-after=60s 43200 \
    bash source/scripts/run_experiment.sh \
    >"$RUN_ROOT/stdout.log" 2>"$RUN_ROOT/stderr.log"
```

Конкретный image закрепляйте digest'ом (`image@sha256:...`), а не тегом.
В archive положите manifest/hash, redacted config, source commit, input manifest,
usage по компонентам, failed-items, cache hashes, результаты baseline и нового
метода. Не кладите prompt/response целиком, если это запрещено политикой данных.

## 7. Запуск существующего baseline этого репозитория

Для текущего HalluGraph baseline не нужно придумывать YAML вручную: submitter
уже проверяет образ, gateway manifest, 3-QA gate и параметры. Команда запуска
находится в [team runbook](vertex-datasphere-team-runbook.md). Агенту разрешено
изменять там размер выборки и CV-параметры только если владелец эксперимента
это явно утвердил.

Проверить статус и скачать результат:

```bash
datasphere --profile default project job get --id <JOB_ID> --format json
datasphere --profile default project job download-files \
  --id <JOB_ID> --with-logs --with-diagnostics --output-dir outputs/<RUN_ID>
```

## 8. Инструкция, которую можно передать агенту

> Работай поверх существующего baseline. Не изменяй его данные, split, метрики
> или cache entries, пока не доказана transport parity. Используй только
> `HALLU_GATEWAY_URL` и DataSphere secret `HALLU_GATEWAY_API_KEY`; значение
> секрета никогда не выводи. Перед дорогим запуском проверь authenticated
> gateway manifest, structured JSON path и cache-only replay на малом
> deterministic наборе. Включай manifest hash в ключи нового кэша. 429/5xx/
> network errors повторяй до wall-time Job; 4xx останавливай с понятной
> диагностикой. В финальный archive положи redacted config, manifest/hash,
> usage, cache hashes, results и logs.

## 9. Чек-лист принятия

- [ ] Нет ключей и Google credentials в git/YAML/log/artifact.
- [ ] Gateway manifest получен с авторизацией и включён в cache identity.
- [ ] LLM adapter — единственное место, которое знает об endpoint/model.
- [ ] Structured output локально проходит JSON Schema validation.
- [ ] Малый probe и cache-only replay успешны.
- [ ] Baseline и новый метод сравниваются парно на одних входах.
- [ ] Временные ошибки повторяются, конфигурационные ошибки падают сразу.
- [ ] Archive позволяет воспроизвести результат и объяснить каждый failed item.
