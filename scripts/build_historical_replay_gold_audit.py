#!/usr/bin/env python3
"""Build a self-contained, post-seal RAGTruth audit page for one prediction archive.

The command is deliberately analysis-only: it validates a sealed archive, joins the
official RAGTruth labels afterwards, and writes all derived artefacts outside the
archive.  It never imports or invokes either detector.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts import RunArchive, sha256_file


METHODS = ("hallugraph", "grapheval")
METHOD_LABELS = {"hallugraph": "HalluGraph", "grapheval": "GraphEval"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def auroc(rows: Iterable[dict[str, Any]]) -> float | None:
    pairs = [(float(row["score"]), int(row["gold"])) for row in rows]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(enumerate(pairs), key=lambda pair: pair[1][0])
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[index][1][0]:
            end += 1
        mean_rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = mean_rank
        index = end
    sum_positive_ranks = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (sum_positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)


def metrics(rows: Iterable[dict[str, Any]], threshold: float) -> dict[str, Any]:
    rows = list(rows)
    decisions = [float(row["score"]) > threshold for row in rows]
    labels = [int(row["gold"]) for row in rows]
    tp = sum(decision and label == 1 for decision, label in zip(decisions, labels))
    fp = sum(decision and label == 0 for decision, label in zip(decisions, labels))
    tn = sum(not decision and label == 0 for decision, label in zip(decisions, labels))
    fn = sum(not decision and label == 1 for decision, label in zip(decisions, labels))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "threshold": threshold,
        "threshold_comparator": ">",
        "n_scored": len(rows),
        "AUROC": auroc(rows),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "F1": safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": None if recall is None or specificity is None else (recall + specificity) / 2,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def choose_f1_threshold(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """Use training scores only. On exact F1 ties prefer recall then lower threshold."""
    if not rows:
        raise ValueError("cannot select a threshold without training score rows")
    candidates = sorted({float(row["score"]) for row in rows} | {0.0, 1.0})
    evaluated = [(candidate, metrics(rows, candidate)) for candidate in candidates]
    chosen = max(
        evaluated,
        key=lambda item: (
            item[1]["F1"] if item[1]["F1"] is not None else -1.0,
            item[1]["recall"] if item[1]["recall"] is not None else -1.0,
            -item[0],
        ),
    )
    return chosen


def compact_components(method: str, row: dict[str, Any]) -> dict[str, Any]:
    components = row.get("components") or {}
    if method == "hallugraph":
        wanted = ("CFI", "EG", "RP", "ungrounded_entities", "unsupported_relations", "aggregation")
        return {key: components.get(key) for key in wanted if key in components}
    triples = []
    for triple in components.get("triples") or []:
        triples.append({
            "id": triple.get("triple_id"),
            "hypothesis": triple.get("verbalized_hypothesis"),
            "p_unsupported": triple.get("p_unsupported"),
            "flagged_at_paper_threshold": triple.get("flagged_at_paper_threshold"),
        })
    return {
        "aggregation": components.get("aggregation"),
        "paper_threshold": components.get("paper_threshold"),
        "paper_threshold_decision": components.get("paper_threshold_decision"),
        "n_triples_total": components.get("n_triples_total"),
        "flagged_unit_ids": row.get("flagged_unit_ids") or components.get("flagged_unit_ids") or [],
        "triples": triples,
    }


def json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def page(payload: dict[str, Any]) -> str:
    data = json_for_script(payload)
    return """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Аудит HalluGraph и GraphEval — 750 RAGTruth</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#182033;--muted:#5f6d85;--line:#dbe2ee;--blue:#155eef;--green:#087443;--red:#b42318;--amber:#a15c00;--violet:#6941c6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1800px;margin:auto;padding:24px}}h1{{font-size:25px;margin:0 0 5px}}h2{{font-size:18px;margin:0 0 10px}}h3{{font-size:15px;margin:0 0 7px}}p{{margin:6px 0}}.muted{{color:var(--muted)}}.warning{{border-left:4px solid var(--amber);padding:10px 12px;background:#fff8e8;margin:16px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:16px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;box-shadow:0 1px 2px #1018280b}}.metric{{font-size:26px;font-weight:700}}.metric-row{{display:grid;grid-template-columns:1.2fr repeat(5,.8fr);gap:8px;align-items:center}}.metric-row>div{{padding:5px}}.metric-header{{font-size:12px;color:var(--muted);font-weight:600}}.controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:16px 0}}label{{display:grid;gap:4px;font-weight:600}}input,select,button{{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink)}}button{{cursor:pointer;font-weight:600}}button:hover{{border-color:var(--blue)}}.table-wrap{{background:#fff;border:1px solid var(--line);border-radius:10px;overflow:auto;max-height:48vh}}table{{border-collapse:collapse;width:100%;font-size:12px}}th{{position:sticky;top:0;background:#eef3fb;z-index:1;text-align:left;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid var(--line);vertical-align:top}}tbody tr{{cursor:pointer}}tbody tr:hover,tbody tr.selected{{background:#eef5ff}}.num{{font-variant-numeric:tabular-nums;text-align:right}}.tag{{display:inline-block;padding:1px 6px;border-radius:999px;font-size:11px;font-weight:700}}.tag.pos{{background:#fdecea;color:var(--red)}}.tag.neg{{background:#e7f6ee;color:var(--green)}}.tag.na{{background:#fff3d6;color:var(--amber)}}.tag.err{{background:#f2eafa;color:var(--violet)}}.detail{{display:grid;grid-template-columns:minmax(330px,1fr) minmax(400px,1.2fr);gap:14px;margin-top:16px}}.text-block{{white-space:pre-wrap;max-height:250px;overflow:auto;background:#fafcff;border:1px solid var(--line);padding:9px;border-radius:6px}}.method-columns{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.method{{padding:10px;border:1px solid var(--line);border-radius:7px}}.good{{color:var(--green);font-weight:700}}.bad{{color:var(--red);font-weight:700}}.graph-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}}.graph-tab.active{{background:var(--blue);color:#fff;border-color:var(--blue)}}.graph{{width:100%;height:370px;border:1px solid var(--line);border-radius:7px;background:#fbfdff}}.edge-table{{max-height:180px;overflow:auto;margin-top:8px}}.edge-table table{{font-size:11px}}.legend{{font-size:12px;color:var(--muted)}}details summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow:auto;max-height:210px;font-size:11px;background:#fafcff;padding:8px;border:1px solid var(--line);border-radius:6px}}@media(max-width:900px){{main{{padding:12px}}.detail,.method-columns{{grid-template-columns:1fr}}.metric-row{{grid-template-columns:1fr 1fr 1fr}}.metric-row .metric-header{{display:none}}.table-wrap{{max-height:42vh}}}}
</style></head><body><main>
<h1>Аудит сравнения HalluGraph и GraphEval</h1>
<p class="muted" id="subtitle"></p><div class="warning" id="warning"></div>
<section class="cards" id="summary"></section>
<section class="card"><h2>Метрики на отложенной тестовой части</h2><p class="muted">Порог для каждого метода выбирается только по обучающей части, максимум F1; затем фиксируется и применяется к тестовой. ROC-AUC не зависит от порога. Строки без численного балла исключены из метрик и показаны отдельно.</p><div id="metrics"></div></section>
<section><div class="controls"><label>Поиск по ID, запросу или ответу<input id="search" type="search" placeholder="например, 12415"></label><label>Часть данных<select id="split"><option value="all">все</option><option value="train">обучающая</option><option value="test">тестовая</option></select></label><label>Показать<select id="outcome"><option value="all">все строки</option><option value="any-error">ошибка хотя бы метода</option><option value="disagree">методы расходятся</option><option value="both-wrong">оба ошиблись</option><option value="hallugraph-wrong">ошибка HalluGraph</option><option value="grapheval-wrong">ошибка GraphEval</option><option value="unscored">без численного балла</option></select></label><label>Сортировка<select id="sort"><option value="id">ID</option><option value="gap">макс. расхождение баллов</option><option value="hallugraph">балл HalluGraph</option><option value="grapheval">балл GraphEval</option></select></label><button id="clear">Сбросить</button></div><p class="muted" id="count"></p><div class="table-wrap"><table><thead><tr><th>ID</th><th>Часть</th><th>Задача</th><th>Эталон</th><th>HalluGraph</th><th>GraphEval</th><th>Итог сравнения</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="detail"><div class="card"><h2 id="detail-title">Выберите строку</h2><div id="detail-meta" class="muted"></div><h3>Запрос</h3><div class="text-block" id="query"></div><h3>Контекст</h3><div class="text-block" id="context"></div><h3>Ответ модели</h3><div class="text-block" id="response"></div></div><div class="card"><h2>Решения и графы</h2><div class="method-columns" id="methods"></div><h3 style="margin-top:14px">Графы текущего примера</h3><div class="graph-tabs"><button class="graph-tab active" data-role="context">Контекст</button><button class="graph-tab" data-role="query">Запрос</button><button class="graph-tab" data-role="response">Ответ</button></div><p class="legend" id="graph-note"></p><svg class="graph" id="graph" role="img" aria-label="Граф сущностей и отношений"></svg><div class="edge-table" id="edges"></div></div></section>
</main><script id="audit-data" type="application/json">__AUDIT_DATA__</script><script>
const D=JSON.parse(document.getElementById('audit-data').textContent), $=id=>document.getElementById(id); let selected=null,role='context';
const n=x=>x==null?'—':Number(x).toFixed(3), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }}[c]));
const tag=(label,kind)=>`<span class="tag ${{kind}}">${{esc(label)}}</span>`;
function outcome(m,r){{if(m.status!=='ok'||m.score==null)return 'без балла'; return (m.decision===Boolean(r.gold))?'верно':'ошибка';}}
function outcomeTag(m,r){{let x=outcome(m,r);return tag(x,x==='верно'?'neg':x==='ошибка'?'pos':'na')}}
function init(){{$('subtitle').textContent=`Запечатанный архив: ${{D.provenance.archive_id}} · ${{D.records.length}} ответов · официальный response.jsonl: SHA-256 ${{D.provenance.responses_sha256.slice(0,16)}}…`; $('warning').innerHTML=D.provenance.warning; renderSummary(); bind(); renderRows();}}
function renderSummary(){{const s=D.summary; $('summary').innerHTML=`<div class="card"><div class="muted">Записей в архиве</div><div class="metric">${{s.n_records}}</div><div class="muted">${{s.n_scored}} с баллами обоих методов</div></div><div class="card"><div class="muted">HalluGraph: покрытие</div><div class="metric">${{(100*s.coverage.hallugraph).toFixed(1)}}%</div><div class="muted">${{s.statuses.hallugraph.ok}} ok, ${{s.statuses.hallugraph.other}} без балла</div></div><div class="card"><div class="muted">GraphEval: покрытие</div><div class="metric">${{(100*s.coverage.grapheval).toFixed(1)}}%</div><div class="muted">${{s.statuses.grapheval.ok}} ok, ${{s.statuses.grapheval.other}} без балла</div></div>`; const rows=D.metrics.map(m=>`<div class="metric-row"><div><b>${{m.label}}</b><div class="muted">порог по train: &gt; ${{n(m.threshold)}}</div></div><div><span class="metric-header">ROC-AUC</span><br><b>${{n(m.test.AUROC)}}</b></div><div><span class="metric-header">F1</span><br><b>${{n(m.test.F1)}}</b></div><div><span class="metric-header">Полнота</span><br><b>${{n(m.test.recall)}}</b></div><div><span class="metric-header">Точность</span><br><b>${{n(m.test.precision)}}</b></div><div><span class="metric-header">TP / FP / TN / FN</span><br><b>${{m.test.confusion_matrix.tp}} / ${{m.test.confusion_matrix.fp}} / ${{m.test.confusion_matrix.tn}} / ${{m.test.confusion_matrix.fn}}</b></div></div>`).join(''); $('metrics').innerHTML=`<div class="metric-row metric-header"><div>Метод</div><div>ROC-AUC</div><div>F1</div><div>Полнота</div><div>Точность</div><div>Матрица ошибок</div></div>${{rows}}`;}}
function bind(){{['search','split','outcome','sort'].forEach(id=>$(id).addEventListener('input',renderRows));$('clear').onclick=()=>{{$('search').value='';$('split').value='all';$('outcome').value='all';$('sort').value='id';renderRows();}};document.querySelectorAll('.graph-tab').forEach(b=>b.onclick=()=>{{role=b.dataset.role;document.querySelectorAll('.graph-tab').forEach(x=>x.classList.toggle('active',x===b));renderGraph();}})}}
function visible(r){{const q=$('search').value.trim().toLowerCase(), split=$('split').value,o=$('outcome').value; const h=r.methods.hallugraph,g=r.methods.grapheval; if(split!=='all'&&r.split!==split)return false;if(q&&!(`${{r.response_id}} ${{r.query}} ${{r.response}}`.toLowerCase().includes(q)))return false;const hw=outcome(h,r)==='ошибка',gw=outcome(g,r)==='ошибка',un=h.score==null||g.score==null;if(o==='any-error'&&!hw&&!gw)return false;if(o==='disagree'&&(h.decision===g.decision||un))return false;if(o==='both-wrong'&&!(hw&&gw))return false;if(o==='hallugraph-wrong'&&!hw)return false;if(o==='grapheval-wrong'&&!gw)return false;if(o==='unscored'&&!un)return false;return true;}}
function renderRows(){{let rows=D.records.filter(visible),sort=$('sort').value;rows.sort((a,b)=>{{if(sort==='id')return Number(a.response_id)-Number(b.response_id);if(sort==='gap')return Math.abs((b.methods.hallugraph.score??-1)-(b.methods.grapheval.score??-1))-Math.abs((a.methods.hallugraph.score??-1)-(a.methods.grapheval.score??-1));return (b.methods[sort].score??-1)-(a.methods[sort].score??-1)}});$('count').textContent=`Показано ${{rows.length}} из ${{D.records.length}}; щёлкните строку, чтобы открыть первичные данные, решения и графы.`;$('rows').innerHTML=rows.map(r=>{{const h=r.methods.hallugraph,g=r.methods.grapheval;let state=h.score==null||g.score==null?tag('без балла','na'):h.decision===g.decision?tag('согласны','neg'):tag('расходятся','err');return `<tr data-id="${{r.response_id}}" class="${{selected?.response_id===r.response_id?'selected':''}}"><td><b>${{r.response_id}}</b></td><td>${{esc(r.split)}}</td><td>${{esc(r.task||'—')}}</td><td>${{r.gold?tag('галлюцинация','pos'):tag('нет','neg')}}</td><td class="num">${{n(h.score)}}<br>${{outcomeTag(h,r)}}</td><td class="num">${{n(g.score)}}<br>${{outcomeTag(g,r)}}</td><td>${{state}}</td></tr>`;}}).join('');document.querySelectorAll('#rows tr').forEach(tr=>tr.onclick=()=>select(D.records.find(r=>r.response_id===tr.dataset.id)));}}
function select(r){{selected=r;$('detail-title').textContent=`Пример ${{r.response_id}}`; $('detail-meta').innerHTML=`${{esc(r.split)}} · ${{esc(r.task||'задача не указана')}} · источник ${{esc(r.source_id)}} · эталон: ${{r.gold?tag('галлюцинация','pos'):tag('без галлюцинации','neg')}}`; $('query').textContent=r.query;$('context').textContent=r.context;$('response').textContent=r.response;$('methods').innerHTML=['hallugraph','grapheval'].map(k=>{{let m=r.methods[k], c=m.components;let extras=k==='hallugraph'?`<div><b>Незаземлённые сущности:</b> ${{esc((c.ungrounded_entities||[]).join(', ')||'нет')}}</div><div><b>Неподдержанные отношения:</b> ${{esc((c.unsupported_relations||[]).map(x=>Array.isArray(x)?x.join(' — '):x).join('; ')||'нет')}}</div>`:`<details><summary>Тройки GraphEval (${{c.triples?.length||0}})</summary><pre>${{esc(JSON.stringify(c.triples||[],null,2))}}</pre></details>`;return `<div class="method"><h3>${{D.labels[k]}}</h3><p>Балл: <b>${{n(m.score)}}</b><br>Порог: &gt; ${{n(m.threshold)}}<br>Решение: ${{m.score==null?tag('нет решения','na'):m.decision?tag('галлюцинация','pos'):tag('нет галлюцинации','neg')}}<br>Сверка: ${{outcomeTag(m,r)}}</p>${{extras}}<details><summary>Все компоненты</summary><pre>${{esc(JSON.stringify(c,null,2))}}</pre></details></div>`;}}).join(''); renderRows();renderGraph();}}
function renderGraph(){{if(!selected)return;let g=selected.graphs[role];$('graph-note').textContent=g?`${{role==='context'?'Контекст':role==='query'?'Запрос':'Ответ'}}: ${{g.entities.length}} сущностей, ${{g.relations.length}} отношений.`:'Граф для этой роли отсутствует.';let svg=$('graph');svg.replaceChildren();$('edges').innerHTML='';if(!g)return;let W=800,H=370,NS='http://www.w3.org/2000/svg',add=(tag,attrs,text)=>{{let e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e);return e;}};svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);let nodes=g.entities, pos={{}};nodes.forEach((name,i)=>{{let a=(2*Math.PI*i/nodes.length)-Math.PI/2;pos[name]=[W/2+Math.cos(a)*(W*.36),H/2+Math.sin(a)*(H*.34)];}});let defs=add('defs',{}),marker=document.createElementNS(NS,'marker');marker.setAttribute('id','arrow');marker.setAttribute('viewBox','0 0 10 10');marker.setAttribute('refX','8');marker.setAttribute('refY','5');marker.setAttribute('markerWidth','6');marker.setAttribute('markerHeight','6');marker.setAttribute('orient','auto-start-reverse');let path=document.createElementNS(NS,'path');path.setAttribute('d','M 0 0 L 10 5 L 0 10 z');path.setAttribute('fill','#64748b');marker.append(path);defs.append(marker);g.relations.forEach(([a,r,b])=>{{if(!pos[a]||!pos[b])return;add('line',{{x1:pos[a][0],y1:pos[a][1],x2:pos[b][0],y2:pos[b][1],stroke:'#94a3b8','stroke-width':'1.2','marker-end':'url(#arrow)'}});let x=(pos[a][0]+pos[b][0])/2,y=(pos[a][1]+pos[b][1])/2;add('text',{{x,y,fill:'#475569','font-size':'10','text-anchor':'middle'}},r);}});nodes.forEach(name=>{{let [x,y]=pos[name];add('circle',{{cx:x,cy:y,r:18,fill:'#e8f0ff',stroke:'#155eef','stroke-width':'1.2'}});add('text',{{x,y:y+3,fill:'#182033','font-size':'10','text-anchor':'middle'}},name.length>18?name.slice(0,17)+'…':name);}});$('edges').innerHTML=`<table><thead><tr><th>Сущность</th><th>Отношение</th><th>Сущность</th></tr></thead><tbody>${{g.relations.map(x=>`<tr><td>${{esc(x[0])}}</td><td>${{esc(x[1])}}</td><td>${{esc(x[2])}}</td></tr>`).join('')}}</tbody></table>`;}}
init();
</script></body></html>""".replace("{{", "{").replace("}}", "}").replace("__AUDIT_DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output_html = args.output_html.resolve()
    output_metrics = args.output_metrics.resolve()
    if archive_dir in output_html.parents or archive_dir in output_metrics.parents:
        parser.error("outputs must be outside the sealed archive")
    archive = RunArchive(archive_dir.parent, archive_dir.name)
    validation = archive.validate()
    if not validation["valid"]:
        raise ValueError(f"prediction archive seal is invalid: {validation['errors']}")
    predictions = read_jsonl(archive_dir / "predictions" / "raw_predictions.jsonl")
    instances = {str(row["response_id"]): row for row in read_jsonl(archive_dir / "instances.no_gold.jsonl")}
    gold_rows = read_jsonl(args.responses.resolve())
    gold = {
        str(row["id"]): {
            "gold": int(bool(row.get("labels") or [])),
            "quality": row.get("quality"),
            "split": row.get("split"),
        }
        for row in gold_rows
    }
    if any(row.get("gold_access_state") != "hidden" for row in predictions):
        raise ValueError("archive predictions do not retain hidden gold state")
    if set(instances) != {str(row.get("response_id")) for row in predictions}:
        raise ValueError("prediction and instance response IDs differ")
    missing = sorted(set(instances) - set(gold))
    if missing:
        raise ValueError(f"official response file lacks {len(missing)} archive response IDs")
    split_mismatches = [response_id for response_id, instance in instances.items() if gold[response_id]["split"] != instance.get("split")]
    if split_mismatches:
        raise ValueError(f"official response and archive split differ for {len(split_mismatches)} IDs")

    graph_by_hash = {row["input_sha256"]: row for row in read_jsonl(archive_dir / "shared_graphs" / "graph_index.jsonl")}
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        method = str(row["method"])
        if method not in METHODS:
            raise ValueError(f"unexpected method {method!r}")
        by_id[str(row["response_id"])][method] = row
    if any(set(by_id[response_id]) != set(METHODS) for response_id in instances):
        raise ValueError("not every response has one prediction from each method")

    score_rows: dict[str, dict[str, list[dict[str, Any]]]] = {method: {"train": [], "test": [], "all": []} for method in METHODS}
    for response_id, instance in instances.items():
        for method in METHODS:
            prediction = by_id[response_id][method]
            if prediction.get("status") == "ok" and prediction.get("raw_score") is not None:
                item = {"score": float(prediction["raw_score"]), "gold": gold[response_id]["gold"]}
                split = str(instance.get("split", ""))
                if split not in ("train", "test"):
                    raise ValueError(f"unexpected split {split!r} for {response_id}")
                score_rows[method][split].append(item)
                score_rows[method]["all"].append(item)

    thresholds: dict[str, float] = {}
    report_methods = []
    for method in METHODS:
        threshold, train_selection = choose_f1_threshold(score_rows[method]["train"])
        thresholds[method] = threshold
        report_methods.append({
            "method": method, "label": METHOD_LABELS[method], "threshold": threshold,
            "selection_split": "train", "selection_objective": "max_F1; ties: max_recall, then lower_threshold",
            "train": train_selection, "test": metrics(score_rows[method]["test"], threshold),
            "all_descriptive_only": metrics(score_rows[method]["all"], threshold),
        })

    records = []
    status_counts = {method: defaultdict(int) for method in METHODS}
    for response_id, instance in sorted(instances.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else pair[0]):
        method_rows = {}
        for method in METHODS:
            prediction = by_id[response_id][method]
            status = str(prediction.get("status"))
            status_counts[method]["ok" if status == "ok" and prediction.get("raw_score") is not None else "other"] += 1
            score = prediction.get("raw_score")
            method_rows[method] = {
                "status": status, "score": None if score is None else float(score), "threshold": thresholds[method],
                "decision": None if score is None else float(score) > thresholds[method],
                "failure": prediction.get("failure"), "components": compact_components(method, prediction),
            }
        graphs = {}
        for role, field in (("context", "context_hash"), ("query", "query_hash"), ("response", "response_hash")):
            graph = graph_by_hash.get(instance[field])
            graphs[role] = None if graph is None else {"entities": graph.get("entities") or [], "relations": graph.get("relations") or []}
        records.append({
            "response_id": response_id, "source_id": instance.get("source_id"), "split": instance.get("split"),
            "task": (instance.get("metadata") or {}).get("task"), "gold": gold[response_id]["gold"], "quality": gold[response_id]["quality"],
            "query": instance.get("query_raw") or "", "context": instance.get("context_raw") or "", "response": instance.get("response_raw") or "",
            "methods": method_rows, "graphs": graphs,
        })

    pair_scored = sum(all(record["methods"][method]["score"] is not None for method in METHODS) for record in records)
    metrics_report = {
        "evaluation_version": "historical-replay-gold-audit-v1", "analysis_only": True,
        "gold_policy": "response_has_at_least_one_official_RAGTruth_label", "threshold_protocol": "choose_max_F1_on_train_then_evaluate_once_on_test",
        "archive_dir": str(archive_dir), "archive_validation": validation, "responses_path": str(args.responses.resolve()),
        "responses_sha256": sha256_file(args.responses.resolve()), "methods": report_methods,
        "coverage": {method: len(score_rows[method]["all"]) / len(records) for method in METHODS},
        "n_records": len(records), "n_pair_scored": pair_scored,
    }
    payload = {
        "labels": METHOD_LABELS, "provenance": {
            "archive_id": archive_dir.name, "responses_sha256": metrics_report["responses_sha256"],
            "warning": "В архиве 749 пар результатов. Завершение исходного облачного задания помечено ошибкой из-за прежнего правила, считавшего <code>empty_graph</code> общей ошибкой; в этой странице строки с <code>empty_graph</code> честно отмечены как «без балла», а целостность запечатанного архива проверена перед расчётом.",
        },
        "summary": {
            "n_records": len(records), "n_scored": pair_scored, "coverage": metrics_report["coverage"],
            "statuses": {method: dict(status_counts[method]) for method in METHODS},
        }, "metrics": report_methods, "records": records,
    }
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page(payload), encoding="utf-8")
    output_metrics.write_text(json.dumps(metrics_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(output_html), "metrics": str(output_metrics), "n_records": len(records), "thresholds": thresholds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
