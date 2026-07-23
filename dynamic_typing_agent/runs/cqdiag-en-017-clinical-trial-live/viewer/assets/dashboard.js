(() => {
  "use strict";

  const data = window.__TYPING_RUN__;
  const grid = document.getElementById("case-grid");
  const empty = document.getElementById("empty-state");
  const search = document.getElementById("case-search");
  const statusFilter = document.getElementById("status-filter");
  const modeFilter = document.getElementById("mode-filter");

  const escapeHtml = (value) => String(value ?? "").replace(
    /[&<>"']/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
  );
  const modeLabel = (mode) => ({
    text: "Текст → KGGen",
    graphs: "Готовые графы",
    auto: "Автоопределение",
    legacy: "Старый формат",
  })[mode] || mode || "Не указан";
  const statusLabel = (status) => ({
    ok: "Готово",
    failed: "Ошибка",
    partial: "Частично",
  })[status] || status || "Неизвестно";

  const cases = Array.isArray(data?.cases) ? data.cases : [];
  document.getElementById("run-title").textContent = data?.run_id || "Результаты запуска";
  document.getElementById("run-description").textContent =
    `${cases.length} ${cases.length === 1 ? "пример" : "примеров"} · единый формат артефактов · без золотых меток`;
  document.getElementById("run-mode").textContent = modeLabel(data?.input?.requested_mode);
  const failedCount = Number(data?.status_counts?.failed || 0);
  const statusPill = document.getElementById("run-status");
  statusPill.textContent = failedCount ? `Ошибок: ${failedCount}` : "Все примеры обработаны";
  statusPill.className = `pill ${failedCount ? "failure" : "success"}`;

  const totals = cases.reduce((acc, item) => {
    const metrics = item.metrics || {};
    acc.entities += Number(metrics.graph_entities || 0);
    acc.types += Number(metrics.types || 0);
    acc.nli += Number(metrics.nli_results || 0);
    return acc;
  }, { entities: 0, types: 0, nli: 0 });
  const summaries = [
    [cases.length, "примеров"],
    [Number(data?.status_counts?.ok || 0), "успешно"],
    [totals.entities, "вершин графа"],
    [totals.nli, "проверок NLI"],
  ];
  document.getElementById("summary-strip").innerHTML = summaries.map(([value, label]) => `
    <div class="summary-item">
      <span class="summary-value">${escapeHtml(value)}</span>
      <span class="summary-label">${escapeHtml(label)}</span>
    </div>
  `).join("");

  function render() {
    const needle = search.value.trim().toLocaleLowerCase();
    const status = statusFilter.value;
    const mode = modeFilter.value;
    const visible = cases.filter((item) => {
      const matchesText = !needle || String(item.case_id).toLocaleLowerCase().includes(needle);
      const matchesStatus = status === "all" || item.status === status;
      const matchesMode = mode === "all" || item.input_mode === mode;
      return matchesText && matchesStatus && matchesMode;
    });

    grid.innerHTML = visible.map((item) => {
      const metrics = item.metrics || {};
      const cardStatus = item.status === "ok" ? "success" : "failure";
      return `
        <a class="case-card" href="${escapeHtml(item.viewer_path)}">
          <div class="case-card-top">
            <div>
              <h3>${escapeHtml(item.case_id)}</h3>
              <p class="mode">${escapeHtml(modeLabel(item.input_mode))}${item.has_answer ? " · с ответом" : " · источник"}</p>
            </div>
            <span class="pill ${cardStatus}">${escapeHtml(statusLabel(item.status))}</span>
          </div>
          <div class="case-metrics">
            <div class="case-metric"><strong>${escapeHtml(metrics.graph_entities || 0)}</strong><span>вершин</span></div>
            <div class="case-metric"><strong>${escapeHtml(metrics.types || 0)}</strong><span>типов</span></div>
            <div class="case-metric"><strong>${escapeHtml(metrics.nli_results || 0)}</strong><span>NLI</span></div>
          </div>
        </a>
      `;
    }).join("");
    empty.hidden = visible.length > 0;
  }

  [search, statusFilter, modeFilter].forEach((control) => {
    control.addEventListener(control.tagName === "INPUT" ? "input" : "change", render);
  });
  render();
})();
