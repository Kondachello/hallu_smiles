export const meta = {
  name: 'hallugraph-case-audit',
  description: 'Per-case HalluGraph error audit in waves, with an aspect registry that grows between waves',
  whenToUse: 'After exporting single-method audit-case projections, to audit each HalluGraph error case.',
  phases: [
    { title: 'Аудит кейсов', detail: 'один субагент на кейс, волнами' },
    { title: 'Реестр аспектов', detail: 'дедуп предложенных аспектов, один писатель на волну' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   caseIds: string[],            // e.g. ["12415", "16264"]
//   projectionDir: string,        // dir with case-<id>.hg.json
//   outputDir: string,            // dir for per-case audits
//   promptPath: string,           // audit system prompt (base checklist)
//   registryJsonl, registryMd, registryLog: string,
//   aspectRegistryCli: string,    // path to aspect_registry.py
//   python: string,               // interpreter for the CLI
//   waveSize?: number,            // default 12
// }
// ---------------------------------------------------------------------------

const cfg = args || {}
const caseIds = cfg.caseIds || []
const waveSize = cfg.waveSize || 12

if (!caseIds.length) throw new Error('args.caseIds is empty — nothing to audit')

const WORKER_SCHEMA = {
  type: 'object',
  required: ['case_id', 'audit_written_to', 'primary_root_cause', 'proposed_aspect'],
  properties: {
    case_id: { type: 'string' },
    audit_written_to: { type: 'string', description: 'absolute path of the markdown audit this agent wrote' },
    primary_root_cause: {
      type: 'object',
      required: ['coarse_class', 'fine_tag', 'confidence'],
      properties: {
        coarse_class: { type: 'string' },
        fine_tag: { type: 'string' },
        confidence: { type: 'number' },
      },
    },
    hallugraph_error_type: { type: 'string', enum: ['FP', 'FN', 'LABEL_AMBIGUITY', 'NOT_CONFIRMED', 'UNDETERMINED'] },
    kggen_contribution: { type: 'string', enum: ['NONE', 'PARTIAL', 'KGGEN_ONLY_OR_DOMINANT'] },
    registry_aspects_considered: {
      type: 'array',
      description: 'aspect_ids from the dynamic registry that this agent evaluated',
      items: { type: 'string' },
    },
    proposed_aspect: {
      type: ['object', 'null'],
      description: 'a diagnostic aspect the base checklist did not ask for; null if nothing genuinely new',
      required: ['title', 'definition', 'how_to_check', 'why_it_matters', 'evidence_in_this_case'],
      properties: {
        title: { type: 'string' },
        definition: { type: 'string' },
        how_to_check: { type: 'string' },
        why_it_matters: { type: 'string' },
        evidence_in_this_case: { type: 'string', description: 'concrete finding in THIS case that motivated it' },
      },
    },
  },
}

const REGISTRAR_SCHEMA = {
  type: 'object',
  required: ['decisions'],
  properties: {
    decisions: {
      type: 'array',
      items: {
        type: 'object',
        required: ['case_id', 'title', 'status', 'reason'],
        properties: {
          case_id: { type: 'string' },
          title: { type: 'string' },
          status: { type: 'string', enum: ['accepted', 'duplicate', 'refinement', 'rejected'] },
          duplicate_of: { type: ['string', 'null'] },
          reason: { type: 'string' },
          aspect_id: { type: ['string', 'null'], description: 'set when status=accepted and the CLI returned an id' },
        },
      },
    },
  },
}

function workerPrompt(caseId, wave) {
  return `Ты — агент-аудитор ошибок HalluGraph. Разбираешь РОВНО ОДИН кейс.

## Что прочитать (именно в этом порядке)

1. **Базовый чек-лист (он же твой рабочий регламент):** \`${cfg.promptPath}\`
   Это основной документ. Следуй его порядку слоёв, правилам маркировки
   (НАБЛЮДЕНИЕ / ВЫВОД / ГИПОТЕЗА / НЕ ПРОВЕРЯЕМО) и обязательному формату ответа.
2. **Динамический реестр аспектов:** \`${cfg.registryMd}\`
   Это аспекты, которые предложили аудиторы на предыдущих кейсах. Реестр может
   быть пуст — тогда просто иди дальше.
3. **Пакет кейса:** \`${cfg.projectionDir}/case-${caseId}.hg.json\`

## Важно про содержимое пакета

В пакете НЕТ вердикта никакого другого детектора — это сделано намеренно.
Не рассуждай о том, что мог бы решить другой метод, и не пытайся его угадать.
Твой предмет — только HalluGraph. Сравнение методов делает отдельная стадия позже.

## Что сделать

**Шаг 1. Базовый разбор.** Пройди кейс по всем слоям базового чек-листа и
подготовь ответ ровно в том формате, который чек-лист требует (разделы 1–12).
Пропусти раздел про вторичный аудит другого метода — его артефактов у тебя нет,
напиши там «артефакты не предоставлены».

**Шаг 2. Реестр аспектов.** Пройдись по каждому аспекту из реестра и укажи,
релевантен ли он этому кейсу. Неприменим — одна строка, без растягивания.

**Шаг 3. Свой аспект.** Теперь главное. Какой аспект я у тебя НЕ спросил, но его
обязательно надо зафиксировать по итогам этого кейса? Это должно быть
наблюдение, которого нет ни в базовом чек-листе, ни в реестре, и которое
обобщается на другие кейсы — не пересказ уже найденной первопричины.
Разбери этот кейс и по своему аспекту тоже, и добавь разбор в отчёт отдельным
разделом «## 13. Новый аспект».

Если действительно нового аспекта нет — верни \`proposed_aspect: null\`.
Это нормальный и ожидаемый исход; пустой аспект ради заполнения поля хуже,
чем честный null, потому что он засоряет реестр для всех следующих агентов.

## Куда писать

Полный отчёт (разделы 1–13) запиши в файл:
\`${cfg.outputDir}/${caseId}.hg.md\`
Начни файл заголовком \`# Кейс ${caseId} — аудит HalluGraph\`.

Отчёт должен быть СУХИМ разбором: что произошло и почему. Без предложений
«давайте внедрим X» — рекомендации собирает отдельная стадия агрегации.

Затем верни structured output. Поле \`audit_written_to\` — путь файла, который ты
записал. Твой финальный текст — это возвращаемое значение, а не письмо человеку.`
}

function registrarPrompt(proposals, wave) {
  return `Ты — регистратор реестра аспектов аудита HalluGraph. Волна ${wave}.

Агенты-аудиторы предложили аспекты после разбора своих кейсов. Твоя задача —
решить, что попадёт в общий реестр, который увидят ВСЕ последующие агенты.

## Текущий реестр

Прочитай \`${cfg.registryMd}\`. Если файла нет или он пуст — реестр пустой.

## Предложения этой волны

\`\`\`json
${JSON.stringify(proposals, null, 2)}
\`\`\`

## Как решать

Для каждого предложения выбери статус:
- \`accepted\` — действительно новый, обобщаемый диагностический аспект;
- \`duplicate\` — по сути уже есть в реестре ИЛИ в базовом чек-листе (\`${cfg.promptPath}\`),
  либо дублирует другое предложение этой же волны (тогда принимай ровно одно);
- \`refinement\` — уточняет существующий аспект, но не тянет на отдельную запись;
- \`rejected\` — не обобщается за пределы одного кейса, или это пересказ
  первопричины, а не способ её обнаружить.

Будь строгим. Реестр читает каждый следующий агент, и раздутый реестр вредит
качеству: он съедает внимание и провоцирует формальные отписки. Планка — аспект
должен менять то, КАК агент смотрит на кейс, а не просто называть найденное.

## Что сделать с принятыми

Для каждого \`accepted\` вызови CLI (Bash), ровно один вызов на аспект:

\`\`\`
${cfg.python} ${cfg.aspectRegistryCli} add \\
  --registry ${cfg.registryJsonl} \\
  --log ${cfg.registryLog} \\
  --markdown ${cfg.registryMd} \\
  --method HalluGraph \\
  --title "<title>" \\
  --definition "<definition>" \\
  --how-to-check "<how_to_check>" \\
  --why-it-matters "<why_it_matters>" \\
  --case "<case_id>" \\
  --agent "wave-${wave}-registrar" \\
  --wave ${wave}
\`\`\`

CLI печатает присвоенный \`aspect_id\` — верни его в поле \`aspect_id\`.

Для НЕ принятых запиши строку в лог, чтобы решение осталось прослеживаемым:
\`\`\`
${cfg.python} - <<'PY'
import json, pathlib
pathlib.Path("${cfg.registryLog}").open("a", encoding="utf-8").write(
    json.dumps({...}, ensure_ascii=False, sort_keys=True) + "\\n")
PY
\`\`\`
где объект содержит: \`status\`, \`title\`, \`proposed_by_case\`, \`wave\`, \`reason\`,
и \`duplicate_of\` если применимо.

Верни structured output со всеми решениями.`
}

// ---------------------------------------------------------------------------

const waves = []
for (let i = 0; i < caseIds.length; i += waveSize) waves.push(caseIds.slice(i, i + waveSize))

log(`${caseIds.length} кейс(ов), ${waves.length} волн(ы) по ${waveSize}`)

const audits = []
const registryDecisions = []

for (let w = 0; w < waves.length; w++) {
  const wave = waves[w]
  const waveNo = w + 1
  const phaseName = `Аудит кейсов`

  // Within a wave the registry is frozen, so every agent in it sees the same
  // registry and no two agents race to write it.
  const results = await parallel(
    wave.map((caseId) => () =>
      agent(workerPrompt(caseId, waveNo), {
        label: `audit:${caseId}`,
        phase: phaseName,
        schema: WORKER_SCHEMA,
      })
    )
  )

  const ok = results.filter(Boolean)
  const lost = wave.length - ok.length
  if (lost > 0) log(`волна ${waveNo}: ${lost} кейс(ов) не вернули результат — см. журнал`)
  audits.push(...ok)

  const proposals = ok
    .filter((r) => r.proposed_aspect)
    .map((r) => ({ case_id: r.case_id, ...r.proposed_aspect }))

  log(`волна ${waveNo}: ${ok.length}/${wave.length} аудитов, ${proposals.length} предложенных аспектов`)

  if (proposals.length) {
    // Single writer per wave — this is what keeps the append-only files consistent.
    const decision = await agent(registrarPrompt(proposals, waveNo), {
      label: `registrar:wave-${waveNo}`,
      phase: 'Реестр аспектов',
      schema: REGISTRAR_SCHEMA,
    })
    if (decision) {
      registryDecisions.push(...decision.decisions)
      const accepted = decision.decisions.filter((d) => d.status === 'accepted')
      log(`волна ${waveNo}: принято аспектов ${accepted.length}/${proposals.length}`)
    }
  }
}

return {
  cases_requested: caseIds.length,
  cases_audited: audits.length,
  audits,
  aspect_decisions: registryDecisions,
  aspects_accepted: registryDecisions.filter((d) => d.status === 'accepted').length,
}
