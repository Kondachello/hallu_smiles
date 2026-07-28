# 03. Live-окружение и воспроизведение

Демо гоняет **гибрид**: типизация (LLM) уходит в gateway-API (`openai/gemini-2.5-flash`),
а NLI считается **локально** на HHEM-2.1-Open (CPU, torch). Конфиг: `config/live-gateway-hhem.yaml`.

## Блокеры, которые пришлось снять (и почему запуск падал раньше)

| # | Симптом | Причина | Обход |
|---|---------|---------|-------|
| 1 | `live backend requires the litellm optional dependency` | `litellm` не установлен | Поставлен в отдельный каталог на D: (см. ниже) |
| 2 | Нет NLI / модель не грузится | Локальный снапшот HHEM не скачан | `scripts/provision_local_resources.py` → `local_resources/` |
| 3 | `transformers` падает: `tokenizers>=0.21,<0.22 … found 0.23.1` | `litellm` тянет `tokenizers 0.23` / `huggingface-hub 1.25`, что несовместимо с `transformers 4.51` (на нём HHEM) | Конфликтующие пакеты удалены из D:-таргета → используются C:-версии |
| 4 | `pip`/кэш/temp падают по месту | Диск **C: заполнен ~100 %** | Всё уведено на D: (`--target`, `HF_HOME`, `TMPDIR`, `PYTHONPYCACHEPREFIX`) |
| 5 | Дикий оверхед на старте, таймауты | `transformers` подтягивает **TensorFlow** | `USE_TF=0` (torch-only) |
| 6 | litellm «не находится» / `source .env` ломается | На Windows-Python `PYTHONPATH` разделяется `;`, а не `:`; в `.env` есть PowerShell-строки с кавычками | Использовать `;` и Windows-пути; из `.env` брать только строки `^HALLU_*=` |

> Замечание про фон: фоновые прогоны здесь **умирают вместе с обрывом сессионного шелла**
> (уведомление «stopped» = процесс реально мёртв). Но иммутабельный кэш (`.cache/…`) переживает
> перезапуск, поэтому возобновление дешёвое. Команда `test` при этом требует **пустую** output-папку —
> для возобновления надо чистить `runs/typing-demo` (кэш API при этом не теряется).

## Провижн ресурсов (один раз)

```bash
cd dynamic_typing_agent
HF_HOME=/d/tmp-typing/hf TMPDIR=/d/tmp-typing \
  python scripts/provision_local_resources.py          # HHEM-2.1-open + flan-t5-base → local_resources/
python scripts/provision_local_resources.py --verify-only
```

Скачивает ~457 МБ в `local_resources/{hhem-2.1-open, flan-t5-base}` (плюс RAGTruth-файлы). Ревизия HHEM
пинится конфигом: `0e7edb3689e710c52ba120086e8f91ea3ee87f23`.

## Установка `litellm` на D: без поломки HHEM-стека

```bash
pip install --target=/d/tmp-typing/site "litellm==1.60.4"
# убрать пакеты, конфликтующие с transformers/HHEM (используем C:-версии):
rm -rf /d/tmp-typing/site/{tokenizers,tokenizers-*.dist-info,\
huggingface_hub,huggingface_hub-*.dist-info,pydantic,pydantic-*.dist-info,\
pydantic_core,pydantic_core-*.dist-info,transformers,transformers-*.dist-info}
```

Проверка коэкзистенции: `import litellm` работает, и одновременно `transformers 4.51.1` +
`tokenizers 0.21.1` + `huggingface_hub 0.30.2` грузят HHEM-токенайзер.

## Команда прогона (воспроизведение)

```bash
cd dynamic_typing_agent

# temp/кэши — на D: (C: полон)
export TMPDIR=/d/tmp-typing TEMP='D:\tmp-typing' TMP='D:\tmp-typing'
export PYTHONPYCACHEPREFIX=/d/tmp-typing/pycache HF_HOME=/d/tmp-typing/hf TRANSFORMERS_CACHE=/d/tmp-typing/hf

# путь к коду агента + litellm с D: (ВНИМАНИЕ: разделитель ';' для Windows-Python)
export PYTHONPATH='src;D:\tmp-typing\site'

# torch-only, оффлайн-хаб, без TF-оверхеда
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

# gateway-креды из .env основного репо (только bash-строки), HHEM — локальный снапшот
while IFS= read -r line; do export "$line"; done \
  < <(grep -E '^HALLU_[A-Z_]+=' /d/RagTruth/hallu_smiles/.env)
export HALLU_HHEM_MODEL_PATH="D:\RagTruth\hallu_smiles\dynamic_typing_agent\local_resources\hhem-2.1-open"

python -m hallugraph_dynamic_typing --config config/live-gateway-hhem.yaml test \
  --input examples/dynamic_typing_20.no_gold.jsonl --input-mode graphs \
  --limit 30 --output runs/typing-demo
```

Флаги: `--input-mode graphs` — берёт **готовые графы** из примеров (KGGen не гоняется);
`--limit 30` покрывает все 20 примеров.

## Тайминг (наблюдаемый, CPU-only)

- разовая загрузка HHEM: ~1.5 мин на старте процесса;
- кэшированный кейс (source из кэша): ~1.2–1.7 мин (answer-типизация + запись артефактов всё равно идут);
- холодный кейс: ~3–7 мин (свежие gateway + NLI вызовы);
- полный прогон 20 примеров «с нуля»: ~1–1.5 часа.

## Входные данные

`examples/dynamic_typing_20.no_gold.jsonl` — **20 примеров**, в каждом по три готовых графа:
`context`, `query`, `answer`. Source-реестр строится из `context`+`query` и замораживается,
затем граф `answer` типизируется против замороженного реестра.
