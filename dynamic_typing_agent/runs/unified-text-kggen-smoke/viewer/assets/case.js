(() => {
  "use strict";

  const data = window.__TYPING_CASE__ || {};
  const registry = data.registry || { types: [], assignments: [], evidence_spans: [], nli_results: [] };
  const annotations = data.annotations || { answer_assignments: [], nli_results: [] };
  const types = Array.isArray(registry.types) ? registry.types : [];
  const assignments = [
    ...(Array.isArray(registry.assignments) ? registry.assignments : []),
    ...(Array.isArray(annotations.answer_assignments) ? annotations.answer_assignments : []),
  ];
  const nliResults = [
    ...(Array.isArray(registry.nli_results) ? registry.nli_results : []),
    ...(Array.isArray(annotations.nli_results) ? annotations.nli_results : []),
  ];
  const spans = new Map((registry.evidence_spans || []).map((span) => [span.span_id, span]));
  const byType = new Map(types.map((type) => [type.type_id, type]));
  const normalize = (value) => String(value ?? "").toLocaleLowerCase().replace(/\s+/g, " ").trim();
  const escapeHtml = (value) => String(value ?? "").replace(
    /[&<>"']/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
  );
  const pretty = (value) => JSON.stringify(value ?? null, null, 2);
  const css = getComputedStyle(document.documentElement);
  const palette = Array.from({ length: 10 }, (_, index) => css.getPropertyValue(`--type-${index}`).trim());
  const typeOrder = new Map(types.map((type, index) => [type.type_id, index]));
  const typeColor = (typeId) => palette[(typeOrder.get(typeId) ?? 0) % palette.length] || "#25755a";
  const contrastText = (color) => {
    const match = /^#?([0-9a-f]{6})$/i.exec(color);
    if (!match) return "#ffffff";
    const value = match[1];
    const channels = [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16) / 255)
      .map((channel) => channel <= .03928 ? channel / 12.92 : ((channel + .055) / 1.055) ** 2.4);
    const luminance = .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
    return luminance > .47 ? "#101613" : "#ffffff";
  };
  const primaryType = (assignment) => {
    const ids = assignment?.type_ids || [];
    return ids.find((id) => id !== "T-ENTITY") || ids[0] || "T-ENTITY";
  };
  const typeNames = (ids) => (ids || []).map((id) => byType.get(id)?.label || id);
  const roleLabel = (role) => ({ context: "Контекст", query: "Запрос", answer: "Ответ" })[role] || role;
  const modeLabel = (mode) => ({ text: "Текст → KGGen", graphs: "Готовые графы", legacy: "Старый формат" })[mode] || mode;
  const verdictLabel = (verdict) => ({ entailed: "подтверждено", neutral: "нейтрально", contradicted: "противоречит" })[verdict] || verdict;
  const evidenceLabel = (level) => ({
    source_entailed: "подтверждено источником",
    definition_only: "основано на определении",
    example_supported: "поддержано примером",
    unknown: "слабое основание",
  })[level] || level;

  const roleInfo = {
    context: {
      label: "Контекст",
      text: data.source?.context_raw || "",
      graph: data.source?.context_graph || { entities: [], relations: [] },
    },
    query: {
      label: "Запрос",
      text: data.source?.query_raw || "",
      graph: data.source?.query_graph || { entities: [], relations: [] },
    },
  };
  if (data.answer) {
    roleInfo.answer = {
      label: "Ответ",
      text: data.answer.response_raw || "",
      graph: data.answer.answer_graph || { entities: [], relations: [] },
    };
  }

  const state = {
    role: Object.keys(roleInfo)[0] || "context",
    entity: null,
    typeId: null,
    inspector: "types",
    eventIndex: null,
  };

  const traceEvents = [
    ...((data.trace?.input_events || []).map((event) => ({ ...event, _phase: "input" }))),
    ...((data.trace?.source_events || []).map((event) => ({ ...event, _phase: "source" }))),
    ...((data.trace?.answer_events || []).map((event) => ({ ...event, _phase: "answer" }))),
  ];

  function assignmentsForSurface(surface, role = null) {
    const key = normalize(surface);
    return assignments.filter((item) =>
      normalize(item.surface_text) === key && (!role || item.graph_role === role)
    );
  }

  function entityTypeIds(surface, role = null) {
    return [...new Set(assignmentsForSurface(surface, role).flatMap((item) => item.type_ids || []))];
  }

  function entityColor(surface, role = null) {
    const typeId = entityTypeIds(surface, role).find((id) => id !== "T-ENTITY")
      || entityTypeIds(surface, role)[0]
      || "T-ENTITY";
    return typeColor(typeId);
  }

  function roleEntities(role) {
    const graph = roleInfo[role]?.graph || {};
    return [...new Set([
      ...(graph.entities || []),
      ...(graph.relations || []).flatMap((relation) => [relation[0], relation[2]]),
    ])].filter(Boolean);
  }

  document.getElementById("case-title").textContent = data.case_id || "Пример";
  document.title = `${data.case_id || "Пример"} · типизация`;
  const chips = [
    [modeLabel(data.input_mode), "quiet"],
    [data.source_status === "ok" && (!data.answer || data.answer_status === "ok") ? "Готово" : "Есть ошибка", data.source_status === "ok" ? "success" : "failure"],
    [`Типов: ${types.length}`, "quiet"],
    [`NLI: ${nliResults.length}`, "quiet"],
  ];
  document.getElementById("case-chips").innerHTML = chips.map(([label, kind]) =>
    `<span class="pill ${kind}">${escapeHtml(label)}</span>`
  ).join("");
  if (data.failure) {
    const banner = document.getElementById("failure-banner");
    banner.hidden = false;
    banner.textContent = `${data.failure.error_type || "Ошибка"}: ${data.failure.message || "обработка не завершена"}`;
  }

  function renderRoleTabs() {
    const tabs = document.getElementById("role-tabs");
    tabs.innerHTML = Object.entries(roleInfo).map(([role, info]) => `
      <button type="button" data-role="${role}" aria-selected="${role === state.role}">
        ${escapeHtml(info.label)}
      </button>
    `).join("");
    tabs.querySelectorAll("[data-role]").forEach((button) => {
      button.addEventListener("click", () => {
        state.role = button.dataset.role;
        state.entity = null;
        state.typeId = null;
        graph.setRole(state.role);
        renderRoleTabs();
        renderText();
        renderInspector();
        renderGraphCaption();
        renderEmptyDetail();
      });
    });
  }

  function sentences(text) {
    const matches = String(text).match(/[^.!?]+[.!?]+|[^.!?]+$/gu);
    return (matches || [String(text)]).map((item) => item.trim()).filter(Boolean);
  }

  function highlightedSentence(text, role) {
    const entities = roleEntities(role).sort((a, b) => b.length - a.length);
    const lower = text.toLocaleLowerCase();
    let cursor = 0;
    let html = "";
    while (cursor < text.length) {
      let match = null;
      for (const entity of entities) {
        if (lower.startsWith(entity.toLocaleLowerCase(), cursor)) {
          match = entity;
          break;
        }
      }
      if (!match) {
        html += escapeHtml(text[cursor]);
        cursor += 1;
        continue;
      }
      const selected = normalize(state.entity) === normalize(match) ? " is-selected" : "";
      html += `<button type="button" class="entity-mention${selected}" data-entity="${escapeHtml(match)}" style="--entity-color:${entityColor(match, role)}">${escapeHtml(text.slice(cursor, cursor + match.length))}</button>`;
      cursor += match.length;
    }
    return html;
  }

  function renderText() {
    const info = roleInfo[state.role];
    document.getElementById("text-title").textContent = info.label;
    const text = document.getElementById("document-text");
    if (!info.text.trim()) {
      text.innerHTML = '<p class="text-empty">Текст для этой части входа отсутствует.</p>';
      return;
    }
    text.innerHTML = sentences(info.text).map((sentence, index) => {
      const selected = state.entity && normalize(sentence).includes(normalize(state.entity)) ? " is-linked" : "";
      return `<div class="sentence-block${selected}"><span class="sentence-number">${String(index + 1).padStart(2, "0")}</span><span>${highlightedSentence(sentence, state.role)}</span></div>`;
    }).join("");
    text.querySelectorAll("[data-entity]").forEach((button) => {
      button.addEventListener("click", () => selectEntity(button.dataset.entity, state.role));
    });
  }

  function typeCount(typeId) {
    return assignments.filter((item) => (item.type_ids || []).includes(typeId)).length;
  }

  function renderTypeTree() {
    const query = normalize(document.getElementById("inspector-search").value);
    const tree = document.getElementById("type-tree");
    const children = new Map();
    for (const type of types) {
      const parents = type.parent_type_ids?.length ? type.parent_type_ids : ["__root__"];
      for (const parent of parents) {
        children.set(parent, [...(children.get(parent) || []), type]);
      }
    }
    const visited = new Set();
    const walk = (parentId, depth) => (children.get(parentId) || [])
      .sort((left, right) => left.label.localeCompare(right.label))
      .map((type) => {
        if (visited.has(type.type_id)) return "";
        visited.add(type.type_id);
        const matches = !query || normalize(`${type.label} ${type.definition}`).includes(query);
        const descendants = walk(type.type_id, depth + 1);
        if (!matches && !descendants) return "";
        const selected = state.typeId === type.type_id ? " is-selected" : "";
        return `
          <button type="button" class="type-row${selected}" data-type="${escapeHtml(type.type_id)}" style="padding-left:${8 + depth * 15}px;--row-color:${typeColor(type.type_id)}">
            <span class="color-dot" style="--dot-color:${typeColor(type.type_id)}"></span>
            <span class="type-row-main">
              <span class="type-label">${escapeHtml(type.label)}</span>
              <span class="type-meta">${typeCount(type.type_id)} назначений · ${escapeHtml(evidenceLabel(type.evidence_level))}</span>
            </span>
          </button>
          ${descendants}
        `;
      }).join("");
    let html = walk("__root__", 0);
    for (const type of types) {
      if (!visited.has(type.type_id)) html += walk(type.type_id, 0);
    }
    tree.innerHTML = html || '<p class="tree-empty">Типы не найдены.</p>';
    tree.querySelectorAll("[data-type]").forEach((button) => {
      button.addEventListener("click", () => selectType(button.dataset.type));
    });
  }

  function renderEntityList() {
    const query = normalize(document.getElementById("inspector-search").value);
    const list = document.getElementById("entity-list");
    const entities = roleEntities(state.role)
      .filter((entity) => !query || normalize(entity).includes(query))
      .sort((left, right) => left.localeCompare(right));
    list.innerHTML = entities.map((entity) => {
      const ids = entityTypeIds(entity, state.role);
      const selected = normalize(state.entity) === normalize(entity) ? " is-selected" : "";
      const color = entityColor(entity, state.role);
      return `
        <button type="button" class="entity-row${selected}" data-entity="${escapeHtml(entity)}" style="--row-color:${color}">
          <span class="color-dot" style="--dot-color:${color}"></span>
          <span class="entity-row-main">
            <span class="entity-label">${escapeHtml(entity)}</span>
            <span class="entity-meta">${escapeHtml(typeNames(ids).join(" · ") || "тип не найден")}</span>
          </span>
        </button>
      `;
    }).join("") || '<p class="tree-empty">Сущности не найдены.</p>';
    list.querySelectorAll("[data-entity]").forEach((button) => {
      button.addEventListener("click", () => selectEntity(button.dataset.entity, state.role));
    });
  }

  function renderInspector() {
    document.querySelectorAll("[data-inspector]").forEach((button) => {
      button.setAttribute("aria-selected", String(button.dataset.inspector === state.inspector));
    });
    document.getElementById("type-tree").hidden = state.inspector !== "types";
    document.getElementById("entity-list").hidden = state.inspector !== "entities";
    renderTypeTree();
    renderEntityList();
  }

  function evidenceChips(ids) {
    return [...new Set(ids || [])].map((id) => {
      const span = spans.get(id);
      return `<button type="button" class="span-chip" data-span="${escapeHtml(id)}">${escapeHtml(span?.text || id)}</button>`;
    }).join("");
  }

  function bindDetailControls() {
    document.querySelectorAll("#detail-body [data-type]").forEach((button) => {
      button.addEventListener("click", () => selectType(button.dataset.type));
    });
    document.querySelectorAll("#detail-body [data-entity]").forEach((button) => {
      button.addEventListener("click", () => selectEntity(button.dataset.entity, button.dataset.role || state.role));
    });
    document.querySelectorAll("#detail-body [data-span]").forEach((button) => {
      button.addEventListener("click", () => {
        const span = spans.get(button.dataset.span);
        if (span?.source_role && roleInfo[span.source_role]) {
          state.role = span.source_role;
          renderRoleTabs();
          renderText();
          graph.setRole(state.role);
          document.getElementById("document-text").scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    });
  }

  function showEntityDetail(surface, role) {
    const matches = assignmentsForSurface(surface, role);
    const ids = [...new Set(matches.flatMap((item) => item.type_ids || []))];
    const relatedNli = nliResults.filter((item) =>
      normalize(item.hypothesis).includes(normalize(surface))
    );
    document.getElementById("detail-title").textContent = surface;
    document.getElementById("detail-body").innerHTML = `
      <div class="detail-section">
        <p class="detail-label">Роль в графе</p>
        <p>${escapeHtml(roleLabel(role))}</p>
      </div>
      <div class="detail-section">
        <p class="detail-label">Назначенные типы</p>
        <div class="detail-types">${ids.map((id) => `<button type="button" class="type-chip" data-type="${escapeHtml(id)}" style="--chip-color:${typeColor(id)}"><span class="color-dot" style="--dot-color:${typeColor(id)}"></span>${escapeHtml(byType.get(id)?.label || id)}</button>`).join("") || "Нет назначения"}</div>
      </div>
      ${matches.map((item) => `
        <div class="detail-section">
          <p class="detail-label">Почему назначено</p>
          <p>${escapeHtml(item.reason)}</p>
          ${item.evidence_span_ids?.length ? `<div class="detail-types">${evidenceChips(item.evidence_span_ids)}</div>` : ""}
        </div>
      `).join("")}
      ${relatedNli.length ? `
        <div class="detail-section">
          <p class="detail-label">Связанные проверки NLI</p>
          ${relatedNli.map((item) => `<p><span class="verdict ${escapeHtml(item.verdict)}">${escapeHtml(verdictLabel(item.verdict))}</span> · ${escapeHtml(item.hypothesis || "")}</p>`).join("")}
        </div>
      ` : ""}
    `;
    bindDetailControls();
  }

  function showTypeDetail(typeId) {
    const type = byType.get(typeId);
    if (!type) return;
    const linked = assignments.filter((item) => (item.type_ids || []).includes(typeId));
    const parents = typeNames(type.parent_type_ids || []);
    const relatedNli = nliResults.filter((item) => item.target_type_id === typeId);
    document.getElementById("detail-title").textContent = type.label;
    document.getElementById("detail-body").innerHTML = `
      <div class="detail-section">
        <p class="detail-label">Определение</p>
        <p>${escapeHtml(type.definition)}</p>
      </div>
      <div class="detail-section">
        <p class="detail-label">Место в иерархии</p>
        <p>${parents.length ? `Родитель: ${escapeHtml(parents.join(" · "))}` : "Корневой тип"}</p>
        <p class="muted">${escapeHtml(evidenceLabel(type.evidence_level))} · статус ${escapeHtml(type.status)}</p>
      </div>
      <div class="detail-section">
        <p class="detail-label">Сущности этого типа</p>
        <div class="detail-types">${linked.map((item) => `<button type="button" class="entity-chip" data-entity="${escapeHtml(item.surface_text)}" data-role="${escapeHtml(item.graph_role)}" style="--chip-color:${typeColor(typeId)}">${escapeHtml(item.surface_text)} · ${escapeHtml(roleLabel(item.graph_role))}</button>`).join("") || "Нет назначений"}</div>
      </div>
      ${type.evidence_span_ids?.length ? `<div class="detail-section"><p class="detail-label">Основания</p><div class="detail-types">${evidenceChips(type.evidence_span_ids)}</div></div>` : ""}
      ${relatedNli.length ? `<div class="detail-section"><p class="detail-label">Проверки, связанные с типом</p>${relatedNli.map((item) => `<p><span class="verdict ${escapeHtml(item.verdict)}">${escapeHtml(verdictLabel(item.verdict))}</span> · ${escapeHtml(item.hypothesis || "")}</p>`).join("")}</div>` : ""}
    `;
    bindDetailControls();
  }

  function renderEmptyDetail() {
    document.getElementById("detail-title").textContent = "Ничего не выбрано";
    document.getElementById("detail-body").innerHTML =
      "Выберите сущность в тексте или графе, тип в словаре либо событие агента.";
  }

  function selectEntity(surface, role = state.role) {
    state.entity = surface;
    state.typeId = null;
    if (roleInfo[role]) state.role = role;
    renderRoleTabs();
    renderText();
    renderInspector();
    renderGraphCaption();
    graph.setRole(state.role, false);
    graph.draw();
    showEntityDetail(surface, state.role);
  }

  function selectType(typeId) {
    state.typeId = typeId;
    state.entity = null;
    renderText();
    renderInspector();
    renderGraphCaption();
    graph.draw();
    showTypeDetail(typeId);
  }

  class CanvasGraph {
    constructor(canvas) {
      this.canvas = canvas;
      this.context = canvas.getContext("2d");
      this.role = null;
      this.nodes = [];
      this.edges = [];
      this.scale = 1;
      this.offsetX = 0;
      this.offsetY = 0;
      this.dragNode = null;
      this.panning = false;
      this.moved = false;
      this.pointerStart = null;
      this.frame = null;
      this.iterations = 0;
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas.parentElement);
      this.bind();
    }

    bind() {
      this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
      this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
      this.canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
      this.canvas.addEventListener("pointercancel", (event) => this.pointerUp(event));
      this.canvas.addEventListener("wheel", (event) => this.zoom(event), { passive: false });
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      this.canvas.width = Math.max(1, Math.round(rect.width * ratio));
      this.canvas.height = Math.max(1, Math.round(rect.height * ratio));
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.width = rect.width;
      this.height = rect.height;
      this.draw();
    }

    setRole(role, reset = true) {
      if (this.role === role && !reset) {
        this.draw();
        return;
      }
      this.role = role;
      const graph = roleInfo[role]?.graph || { entities: [], relations: [] };
      const names = roleEntities(role);
      const previous = new Map(this.nodes.map((node) => [node.id, node]));
      this.nodes = names.map((name, index) => {
        const old = previous.get(name);
        const angle = (Math.PI * 2 * index) / Math.max(1, names.length);
        return {
          id: name,
          x: old?.x ?? Math.cos(angle) * Math.min(220, 55 + names.length * 14),
          y: old?.y ?? Math.sin(angle) * Math.min(170, 45 + names.length * 11),
          vx: 0,
          vy: 0,
          radius: 25,
        };
      });
      const ids = new Map(this.nodes.map((node) => [node.id, node]));
      this.edges = (graph.relations || []).map((relation, index) => ({
        id: `${index}:${relation.join("|")}`,
        source: ids.get(relation[0]),
        target: ids.get(relation[2]),
        label: relation[1],
      })).filter((edge) => edge.source && edge.target);
      if (reset) this.resetView();
      this.iterations = 0;
      this.startSimulation();
      document.getElementById("canvas-empty").hidden = this.nodes.length > 0;
    }

    resetView() {
      this.scale = 1;
      this.offsetX = 0;
      this.offsetY = 0;
      this.draw();
    }

    startSimulation() {
      if (this.frame) cancelAnimationFrame(this.frame);
      const run = () => {
        this.step();
        this.draw();
        this.iterations += 1;
        if (this.iterations < 180) this.frame = requestAnimationFrame(run);
      };
      this.frame = requestAnimationFrame(run);
    }

    step() {
      const repulsion = 1900;
      for (let i = 0; i < this.nodes.length; i += 1) {
        for (let j = i + 1; j < this.nodes.length; j += 1) {
          const left = this.nodes[i];
          const right = this.nodes[j];
          let dx = right.x - left.x;
          let dy = right.y - left.y;
          const distance2 = Math.max(80, dx * dx + dy * dy);
          const force = repulsion / distance2;
          const distance = Math.sqrt(distance2);
          dx /= distance;
          dy /= distance;
          left.vx -= dx * force;
          left.vy -= dy * force;
          right.vx += dx * force;
          right.vy += dy * force;
        }
      }
      for (const edge of this.edges) {
        const dx = edge.target.x - edge.source.x;
        const dy = edge.target.y - edge.source.y;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const force = (distance - 145) * .004;
        edge.source.vx += (dx / distance) * force;
        edge.source.vy += (dy / distance) * force;
        edge.target.vx -= (dx / distance) * force;
        edge.target.vy -= (dy / distance) * force;
      }
      for (const node of this.nodes) {
        if (node === this.dragNode) continue;
        node.vx += -node.x * .0009;
        node.vy += -node.y * .0009;
        node.vx *= .88;
        node.vy *= .88;
        node.x += node.vx;
        node.y += node.vy;
      }
    }

    worldToScreen(node) {
      return {
        x: this.width / 2 + this.offsetX + node.x * this.scale,
        y: this.height / 2 + this.offsetY + node.y * this.scale,
      };
    }

    screenToWorld(x, y) {
      return {
        x: (x - this.width / 2 - this.offsetX) / this.scale,
        y: (y - this.height / 2 - this.offsetY) / this.scale,
      };
    }

    hitNode(x, y) {
      return [...this.nodes].reverse().find((node) => {
        const point = this.worldToScreen(node);
        return Math.hypot(x - point.x, y - point.y) <= node.radius * this.scale + 7;
      });
    }

    pointerPosition(event) {
      const rect = this.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    pointerDown(event) {
      const point = this.pointerPosition(event);
      this.pointerStart = point;
      this.moved = false;
      this.dragNode = this.hitNode(point.x, point.y);
      this.panning = !this.dragNode;
      this.canvas.classList.add("is-dragging");
      this.canvas.setPointerCapture(event.pointerId);
    }

    pointerMove(event) {
      if (!this.pointerStart) return;
      const point = this.pointerPosition(event);
      const dx = point.x - this.pointerStart.x;
      const dy = point.y - this.pointerStart.y;
      if (Math.hypot(dx, dy) > 3) this.moved = true;
      if (this.dragNode) {
        const world = this.screenToWorld(point.x, point.y);
        this.dragNode.x = world.x;
        this.dragNode.y = world.y;
        this.dragNode.vx = 0;
        this.dragNode.vy = 0;
      } else if (this.panning) {
        this.offsetX += dx;
        this.offsetY += dy;
      }
      this.pointerStart = point;
      this.draw();
    }

    pointerUp(event) {
      const point = this.pointerPosition(event);
      const clicked = !this.moved ? this.hitNode(point.x, point.y) : null;
      this.dragNode = null;
      this.panning = false;
      this.pointerStart = null;
      this.canvas.classList.remove("is-dragging");
      if (clicked) selectEntity(clicked.id, this.role);
    }

    zoom(event) {
      event.preventDefault();
      const point = this.pointerPosition(event);
      const before = this.screenToWorld(point.x, point.y);
      const factor = event.deltaY < 0 ? 1.12 : .89;
      this.scale = Math.max(.45, Math.min(2.6, this.scale * factor));
      const after = this.worldToScreen(before);
      this.offsetX += point.x - after.x;
      this.offsetY += point.y - after.y;
      this.draw();
    }

    drawArrow(source, target, label, dimmed) {
      const context = this.context;
      const angle = Math.atan2(target.y - source.y, target.x - source.x);
      const startX = source.x + Math.cos(angle) * 29 * this.scale;
      const startY = source.y + Math.sin(angle) * 29 * this.scale;
      const endX = target.x - Math.cos(angle) * 31 * this.scale;
      const endY = target.y - Math.sin(angle) * 31 * this.scale;
      context.globalAlpha = dimmed ? .16 : .58;
      context.strokeStyle = css.getPropertyValue("--line-strong").trim();
      context.fillStyle = css.getPropertyValue("--muted").trim();
      context.lineWidth = 1.25;
      context.beginPath();
      context.moveTo(startX, startY);
      context.lineTo(endX, endY);
      context.stroke();
      context.beginPath();
      context.moveTo(endX, endY);
      context.lineTo(endX - Math.cos(angle - .5) * 8, endY - Math.sin(angle - .5) * 8);
      context.lineTo(endX - Math.cos(angle + .5) * 8, endY - Math.sin(angle + .5) * 8);
      context.closePath();
      context.fill();
      context.font = "11px Inter, Segoe UI, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "bottom";
      const middleX = (startX + endX) / 2;
      const middleY = (startY + endY) / 2;
      const width = context.measureText(label).width + 9;
      context.fillStyle = css.getPropertyValue("--surface-raised").trim();
      context.globalAlpha = dimmed ? .25 : .92;
      context.fillRect(middleX - width / 2, middleY - 15, width, 17);
      context.fillStyle = css.getPropertyValue("--muted").trim();
      context.fillText(label, middleX, middleY - 2);
      context.globalAlpha = 1;
    }

    drawNode(node) {
      const context = this.context;
      const point = this.worldToScreen(node);
      const ids = entityTypeIds(node.id, this.role);
      const selected = normalize(state.entity) === normalize(node.id);
      const typeSelected = state.typeId && ids.includes(state.typeId);
      const dimmed = (state.entity && !selected) || (state.typeId && !typeSelected);
      const color = entityColor(node.id, this.role);
      const radius = node.radius * this.scale;
      context.globalAlpha = dimmed ? .22 : 1;
      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      context.lineWidth = selected || typeSelected ? 5 : 2;
      context.strokeStyle = selected || typeSelected
        ? css.getPropertyValue("--ink").trim()
        : css.getPropertyValue("--surface").trim();
      context.stroke();
      context.globalAlpha = dimmed ? .28 : 1;
      context.fillStyle = contrastText(color);
      context.font = `${Math.max(10, Math.min(13, 11.5 * this.scale))}px Inter, Segoe UI, sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      const words = node.id.split(/\s+/);
      const lines = words.length > 2
        ? [words.slice(0, Math.ceil(words.length / 2)).join(" "), words.slice(Math.ceil(words.length / 2)).join(" ")]
        : [node.id];
      lines.slice(0, 2).forEach((line, index) => {
        const clipped = line.length > 18 ? `${line.slice(0, 16)}…` : line;
        context.fillText(clipped, point.x, point.y + (index - (lines.length - 1) / 2) * 13);
      });
      context.globalAlpha = 1;
    }

    draw() {
      if (!this.context || !this.width || !this.height) return;
      this.context.clearRect(0, 0, this.width, this.height);
      for (const edge of this.edges) {
        const source = this.worldToScreen(edge.source);
        const target = this.worldToScreen(edge.target);
        const dimmed = state.entity
          && normalize(state.entity) !== normalize(edge.source.id)
          && normalize(state.entity) !== normalize(edge.target.id);
        this.drawArrow(source, target, edge.label, dimmed);
      }
      for (const node of this.nodes) this.drawNode(node);
    }
  }

  const graph = new CanvasGraph(document.getElementById("knowledge-graph"));
  document.getElementById("reset-graph").addEventListener("click", () => graph.resetView());

  function renderGraphCaption() {
    const used = [...new Set(roleEntities(state.role).flatMap((entity) => entityTypeIds(entity, state.role)))];
    const caption = document.getElementById("graph-caption");
    caption.innerHTML = used.length
      ? used.map((id) => `<button type="button" class="type-chip" data-type="${escapeHtml(id)}" style="--chip-color:${typeColor(id)}"><span class="color-dot" style="--dot-color:${typeColor(id)}"></span>${escapeHtml(byType.get(id)?.label || id)}</button>`).join("")
      : "<span>Для вершин этой части входа типы не найдены.</span>";
    caption.querySelectorAll("[data-type]").forEach((button) => {
      button.addEventListener("click", () => selectType(button.dataset.type));
    });
  }

  const eventLabels = {
    detect_input_mode: ["Определён формат входа", "Подготовка"],
    load_supplied_graph: ["Загружен готовый граф", "Подготовка"],
    kggen_extract_graph: ["KGGen извлёк граф из текста", "KGGen"],
    prepare_case_input: ["Не удалось подготовить вход", "Ошибка"],
    validate_source: ["Проверен исходный вход", "Система"],
    source_cache: ["Проверен кэш реестра", "Система"],
    segment_source: ["Текст разбит на доказательные фрагменты", "Система"],
    schema_overview: ["Модель предложила обзор возможных типов", "Модель"],
    build_entity_profiles: ["Собраны профили сущностей", "Система"],
    entity_type_decision: ["Модель выбрала тип сущности", "Модель"],
    nli_verify_source: ["NLI проверил назначение типа", "NLI"],
    commit_entity_type: ["Тип закреплён за сущностью", "Решение"],
    registry_consistency_review: ["Модель проверила словарь и иерархию", "Модель"],
    nli_verify_hierarchy: ["NLI проверил иерархию или слияние", "NLI"],
    derive_registry: ["Собран итоговый словарь", "Решение"],
    freeze_registry: ["Словарь заморожен", "Граница"],
    validate_answer: ["Проверен вход ответа", "Система"],
    build_answer_profiles: ["Собраны сущности ответа", "Система"],
    answer_typing: ["Модель выбрала тип сущности ответа", "Модель"],
    nli_verify_answer: ["NLI проверил тип ответа", "NLI"],
    annotate_answer: ["Сформированы назначения ответа", "Решение"],
    nli_answer: ["Собраны проверки ответа", "NLI"],
    emit_answer: ["Результат ответа сохранён", "Граница"],
  };

  function eventIdentity(event) {
    return eventLabels[event.node] || [event.node || "Событие", "Система"];
  }

  function eventSummary(event) {
    const inputs = event.inputs || {};
    const outputs = event.outputs || {};
    const surface = inputs.surface_text
      || inputs.entity_profile?.surface_text
      || outputs.surface_text;
    const response = outputs.response || {};
    if (surface) return String(surface);
    if (response.surface_text) return String(response.surface_text);
    if (response.new_type?.label) return `новый тип: ${response.new_type.label}`;
    if (Array.isArray(outputs.final_type_ids)) return typeNames(outputs.final_type_ids).join(" · ");
    if (Array.isArray(outputs.results) && outputs.results.length) {
      return outputs.results.map((item) => verdictLabel(item.verdict)).join(" · ");
    }
    if (outputs.registry?.types) return `${outputs.registry.types.length} типов`;
    if (outputs.profile_count !== undefined) return `${outputs.profile_count} профилей`;
    if (outputs.assignment_count !== undefined) return `${outputs.assignment_count} назначений`;
    return event._phase === "answer" ? "этап ответа" : "этап источника";
  }

  function eventSearchText(event) {
    const identity = eventIdentity(event);
    return normalize(`${identity.join(" ")} ${eventSummary(event)} ${pretty(event)}`);
  }

  function renderTimeline() {
    const phase = document.getElementById("trace-phase").value;
    const query = normalize(document.getElementById("trace-search").value);
    const timeline = document.getElementById("timeline");
    const visible = traceEvents.map((event, index) => ({ event, index })).filter(({ event }) =>
      (phase === "all" || event._phase === phase)
      && (!query || eventSearchText(event).includes(query))
    );
    timeline.innerHTML = visible.map(({ event, index }) => {
      const [title, kind] = eventIdentity(event);
      return `
        <button type="button" class="timeline-item${state.eventIndex === index ? " is-selected" : ""}" data-event="${index}">
          <span class="timeline-step">${index + 1}</span>
          <span class="timeline-copy">
            <span class="timeline-title">${escapeHtml(title)}</span>
            <span class="timeline-summary">${escapeHtml(eventSummary(event))}</span>
            <span class="timeline-kind">${escapeHtml(kind)} · ${escapeHtml(event._phase === "answer" ? "ответ" : event._phase === "input" ? "подготовка" : "источник")}</span>
          </span>
        </button>
      `;
    }).join("") || '<p class="tree-empty">События не найдены.</p>';
    timeline.querySelectorAll("[data-event]").forEach((button) => {
      button.addEventListener("click", () => selectEvent(Number(button.dataset.event)));
    });
  }

  function modelPayload(event) {
    const outputs = event.outputs || {};
    if (outputs.response !== undefined) return outputs.response;
    if (outputs.overview !== undefined) return outputs.overview;
    if (outputs.proposals !== undefined) return outputs.proposals;
    return null;
  }

  function nliPayload(event) {
    const outputs = event.outputs || {};
    if (Array.isArray(outputs.results)) return outputs.results;
    if (outputs.result) return [outputs.result];
    return [];
  }

  function eventNarrative(event) {
    const inputs = event.inputs || {};
    const outputs = event.outputs || {};
    const response = outputs.response || {};
    const surface = inputs.surface_text
      || inputs.entity_profile?.surface_text
      || response.surface_text
      || "";
    if (event.node === "detect_input_mode") {
      return outputs.selected_mode === "text"
        ? "Вход содержит текст без готового графа, поэтому перед типизацией выбран KGGen."
        : "Вход уже содержит графы; повторное извлечение KGGen не требуется.";
    }
    if (event.node === "kggen_extract_graph") {
      return `KGGen построил граф роли «${roleLabel(outputs.role)}»: ${outputs.entity_count || 0} вершин и ${outputs.relation_count || 0} связей.`;
    }
    if (event.node === "load_supplied_graph") {
      return `Готовый граф роли «${roleLabel(outputs.role)}» принят без повторного извлечения: ${outputs.entity_count || 0} вершин и ${outputs.relation_count || 0} связей.`;
    }
    if (event.node === "entity_type_decision") {
      if (response.new_type) return `Сущность «${surface || response.entity_id}» породила новый тип «${response.new_type.label}».`;
      return `Для сущности «${surface || response.entity_id}» модель выбрала существующий тип: ${typeNames(response.selected_type_ids || []).join(" · ") || "не указан"}.`;
    }
    if (event.node === "commit_entity_type") {
      return `Назначение сохранено: ${surface || "сущность"} → ${typeNames(outputs.final_type_ids || []).join(" · ") || "entity"}.`;
    }
    if (event.node === "registry_consistency_review") {
      const proposals = response.proposals || outputs.proposals || [];
      return proposals.length
        ? `Модель предложила ${proposals.length} изменений словаря: родительские связи или слияния.`
        : "Модель не предложила менять словарь или иерархию.";
    }
    if (event.node === "nli_verify_hierarchy") {
      const proposal = inputs.proposal || {};
      const first = byType.get(proposal.first_type_id)?.label || proposal.first_type_id || "первый тип";
      const second = byType.get(proposal.second_type_id)?.label || proposal.second_type_id || "второй тип";
      if (outputs.accepted && proposal.action === "merge") {
        return `Типы «${first}» и «${second}» склеились после двух подтверждающих проверок.`;
      }
      if (outputs.accepted && proposal.action === "child_of") {
        return `Тип «${first}» стал дочерним для «${second}» после подтверждения NLI.`;
      }
      return `Изменение «${first}» ↔ «${second}» отклонено: требуемого подтверждения NLI нет.`;
    }
    if (event.node === "freeze_registry") {
      return `Словарь закрыт для изменений: ${outputs.registry?.types?.length || types.length} окончательных типов.`;
    }
    if (event.node === "answer_typing") {
      return `Для сущности ответа «${surface || response.entity_id}» выбраны только типы из замороженного словаря.`;
    }
    if (event.node?.startsWith("nli_")) {
      return "NLI сопоставил гипотезу с исходными доказательствами и вернул один из трёх исходов.";
    }
    if (event.node === "schema_overview") {
      return "Модель составила предварительную карту категорий. Эти предложения ещё не являются типами словаря.";
    }
    if (event.node === "derive_registry") {
      return "Все сущности получили типы; после проверки структуры собран итоговый словарь.";
    }
    return eventIdentity(event)[0] + ".";
  }

  function selectEvent(index) {
    state.eventIndex = index;
    renderTimeline();
    const event = traceEvents[index];
    const [title, kind] = eventIdentity(event);
    const model = modelPayload(event);
    const nli = nliPayload(event);
    const detail = document.getElementById("event-detail");
    detail.innerHTML = `
      <div class="event-heading">
        <div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(eventSummary(event))}</p></div>
        <span class="pill quiet">${escapeHtml(kind)}</span>
      </div>
      <div class="event-block emphasis">
        <p class="detail-label">Что произошло</p>
        <p>${escapeHtml(eventNarrative(event))}</p>
      </div>
      ${model !== null ? `
        <div class="event-block">
          <p class="detail-label">Модель ответила / предложила</p>
          <pre>${escapeHtml(pretty(model))}</pre>
        </div>
      ` : ""}
      ${nli.length ? `
        <div class="event-block">
          <p class="detail-label">Решение NLI</p>
          ${nli.map((item) => `
            <p><span class="verdict ${escapeHtml(item.verdict)}">${escapeHtml(verdictLabel(item.verdict))}</span> · ${escapeHtml(item.hypothesis || "")}</p>
            <p class="muted">${escapeHtml(item.rationale || "")}</p>
          `).join("")}
        </div>
      ` : ""}
      <details>
        <summary>Что получила эта стадия</summary>
        <pre>${escapeHtml(pretty(event.inputs || {}))}</pre>
      </details>
      <details>
        <summary>Полный выход стадии</summary>
        <pre>${escapeHtml(pretty(event.outputs || {}))}</pre>
      </details>
      <details>
        <summary>Полная запись события</summary>
        <pre>${escapeHtml(pretty(event))}</pre>
      </details>
    `;
  }

  document.querySelectorAll("[data-inspector]").forEach((button) => {
    button.addEventListener("click", () => {
      state.inspector = button.dataset.inspector;
      document.getElementById("inspector-search").value = "";
      renderInspector();
    });
  });
  document.getElementById("inspector-search").addEventListener("input", renderInspector);
  document.getElementById("trace-phase").addEventListener("change", renderTimeline);
  document.getElementById("trace-search").addEventListener("input", renderTimeline);

  renderRoleTabs();
  renderText();
  renderInspector();
  graph.setRole(state.role);
  renderGraphCaption();
  renderTimeline();
})();
