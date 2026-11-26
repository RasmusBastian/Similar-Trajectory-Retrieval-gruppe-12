// Demo UI: pick dataset, model, ground-truth measure, query trajectory and k,
// then show the model's top-k and the true top-k on a map and in two lists.
"use strict";

const $ = (id) => document.getElementById(id);

const COLORS = { query: "#111827", both: "#7c3aed", model: "#2563eb", truth: "#16a34a" };

const state = {
  options: null,        // from GET /api/options
  queryIds: [],         // valid query ids for the chosen dataset + measure
  layers: new Map(),    // trajectory idx -> { line, category }
  queryLayer: null,
};

// ---------- API ----------

/** GET a JSON endpoint; throws an Error with the server's message on failure. */
async function api(path) {
  const response = await fetch(path);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

// ---------- Map ----------

const map = L.map("map", { preferCanvas: true }).setView([39.9, 116.4], 11);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

function clearMap() {
  for (const { line } of state.layers.values()) line.remove();
  state.layers.clear();
  state.queryLayer?.remove();
  state.queryLayer = null;
}

/**
 * Puts every result trajectory in one of three groups:
 * "both" (in model and ground truth top-k), "model" (model only) or "truth" (ground truth only).
 */
function categorize(result) {
  const modelIds = new Set(result.modelTopK.map((r) => r.idx));
  const truthIds = new Set(result.groundTruthTopK.map((r) => r.idx));
  const category = new Map();
  for (const id of modelIds) category.set(id, truthIds.has(id) ? "both" : "model");
  for (const id of truthIds) if (!modelIds.has(id)) category.set(id, "truth");
  return category;
}

function drawResult(result, category) {
  clearMap();
  const all = [...result.modelTopK, ...result.groundTruthTopK];
  const seen = new Set();
  for (const item of all) {
    if (seen.has(item.idx)) continue;
    seen.add(item.idx);
    const cat = category.get(item.idx);
    const line = L.polyline(item.path, { color: COLORS[cat], weight: 3, opacity: 0.65 })
      .bindTooltip(`Trajectory ${item.idx}`)
      .on("mouseover", () => highlight(item.idx, true))
      .on("mouseout", () => highlight(item.idx, false))
      .addTo(map);
    state.layers.set(item.idx, { line, category: cat });
  }
  state.queryLayer = L.polyline(result.query.path, { color: COLORS.query, weight: 6, opacity: 0.9 })
    .bindTooltip(`Query ${result.query.idx}`)
    .addTo(map);
  applyVisibility();

  const bounds = L.latLngBounds(result.query.path);
  for (const { line } of state.layers.values()) bounds.extend(line.getBounds());
  map.fitBounds(bounds, { padding: [30, 30] });
}

/** Emphasises one trajectory on the map and in both lists. */
function highlight(idx, on) {
  const layer = state.layers.get(idx);
  if (layer) {
    layer.line.setStyle({ weight: on ? 6 : 3, opacity: on ? 1 : 0.65 });
    if (on) layer.line.bringToFront();
    state.queryLayer?.bringToFront();
  }
  document.querySelectorAll(`li[data-idx="${idx}"]`).forEach((li) => li.classList.toggle("active", on));
}

function applyVisibility() {
  const visible = { both: $("show-both").checked, model: $("show-model").checked, truth: $("show-truth").checked };
  for (const { line, category } of state.layers.values()) {
    if (visible[category]) line.addTo(map);
    else line.remove();
  }
}

// ---------- Lists and summary ----------

function renderList(listEl, items, category, unit) {
  listEl.replaceChildren(...items.map((item) => {
    const li = document.createElement("li");
    li.dataset.idx = item.idx;
    li.className = category.get(item.idx);
    li.innerHTML = `<span class="rank">${item.rank}</span><span>#${item.idx}</span>` +
      `<span class="dist">${formatDistance(item.distance, unit)}</span>`;
    li.addEventListener("mouseenter", () => highlight(item.idx, true));
    li.addEventListener("mouseleave", () => highlight(item.idx, false));
    li.addEventListener("click", () => {
      const layer = state.layers.get(item.idx);
      if (layer) map.fitBounds(layer.line.getBounds().extend(state.queryLayer.getBounds()), { padding: [30, 30] });
    });
    return li;
  }));
}

/**
 * Ground truth is in metres: "850 m" or "2.35 km".
 * Model (L1) distances have no unit and their size depends on the model (TSMini ~2, TrajCL ~0.03),
 * so we show 3 significant digits: 1.85, 0.0234, 12.3.
 */
function formatDistance(value, unit) {
  if (unit === "m") return value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${Math.round(value)} m`;
  return Number(value.toPrecision(3)).toString();
}

function renderResult(result) {
  const category = categorize(result);
  const modelName = displayName(state.options.models, result.model);
  const measureName = displayName(state.options.measures, result.measure);

  $("overlap-number").textContent = `${result.overlap.hits}/${result.overlap.k}`;
  $("overlap-text").textContent =
    `of the true top-${result.overlap.k} (${measureName}) found by ${modelName} · recall@${result.overlap.k} = ` +
    `${Math.round(result.overlap.recall * 100)}%`;
  $("model-title").textContent = `${modelName} (L1)`;
  $("truth-title").textContent = `${measureName} (true)`;
  $("checkpoint").textContent =
    `Query #${result.query.idx} (${result.query.split} split) · checkpoint ${result.checkpoint}`;

  renderList($("model-list"), result.modelTopK, category, "l1");
  renderList($("truth-list"), result.groundTruthTopK, category, "m");
  $("results").hidden = false;
  drawResult(result, category);
}

// ---------- Form ----------

const displayName = (list, id) => list.find((x) => x.id === id)?.displayName ?? id;

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function fillSelect(select, items, keep) {
  select.replaceChildren(...items.map(({ value, label, disabled }) => {
    const option = new Option(label, value);
    option.disabled = !!disabled;
    return option;
  }));
  const enabled = items.filter((i) => !i.disabled).map((i) => i.value);
  select.value = enabled.includes(keep) ? keep : (enabled[0] ?? "");
}

function currentDataset() {
  return state.options.datasets.find((d) => d.name === $("dataset").value);
}

/** Enables only the models and measures that have data for the chosen dataset. */
function onDatasetChange(preferred = {}) {
  const dataset = currentDataset();
  fillSelect($("model"), state.options.models.map((m) => ({
    value: m.id,
    label: !m.implemented ? `${m.displayName} (coming soon)`
      : dataset?.models.includes(m.id) ? m.displayName : `${m.displayName} (no data)`,
    disabled: !m.implemented || !dataset?.models.includes(m.id),
  })), preferred.model ?? $("model").value);
  fillSelect($("measure"), state.options.measures.map((m) => ({
    value: m.id,
    label: dataset?.measures.includes(m.id) ? m.displayName : `${m.displayName} (not computed)`,
    disabled: !dataset?.measures.includes(m.id),
  })), preferred.measure ?? $("measure").value);
  return loadQueryIds();
}

async function loadQueryIds() {
  const dataset = $("dataset").value;
  const measure = $("measure").value;
  state.queryIds = [];
  $("query-hint").textContent = "";
  if (!dataset || !measure) return;
  const { ids } = await api(`/api/datasets/${encodeURIComponent(dataset)}/queries?measure=${encodeURIComponent(measure)}`);
  state.queryIds = ids;
  $("query-hint").textContent = ids.length
    ? `${ids.length} trajectories can be used as query (ids ${ids[0]}–${ids[ids.length - 1]}).`
    : "No trajectories with ground truth for this choice.";
  if (!ids.includes(Number($("query").value)) && ids.length) $("query").value = ids[0];
}

function pickRandomQuery() {
  if (!state.queryIds.length) return;
  $("query").value = state.queryIds[Math.floor(Math.random() * state.queryIds.length)];
}

async function search(event) {
  event?.preventDefault();
  const params = new URLSearchParams({
    dataset: $("dataset").value,
    model: $("model").value,
    measure: $("measure").value,
    query: $("query").value,
    k: $("k").value,
  });
  history.replaceState(null, "", `?${params}`); // so a demo link reproduces the search
  $("search").disabled = true;
  setStatus("Searching…");
  try {
    const started = performance.now();
    const result = await api(`/api/search?${params}`);
    renderResult(result);
    setStatus(`Done in ${Math.round(performance.now() - started)} ms.`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    $("search").disabled = false;
  }
}

async function init() {
  setStatus("Loading…");
  try {
    state.options = await api("/api/options");
  } catch (error) {
    setStatus(`Could not reach the API: ${error.message}`, true);
    return;
  }

  const { datasets, k } = state.options;
  if (!datasets.length) {
    setStatus("The database has no datasets yet. Import TSMini/TrajCL data first (see README).", true);
    return;
  }

  // Start from the URL (?dataset=...&model=...) if the page was opened from a shared link.
  const url = new URLSearchParams(location.search);
  fillSelect($("dataset"), datasets.map((d) => ({ value: d.name, label: `${d.name} (${d.trajectoryCount.toLocaleString()} trajectories)` })), url.get("dataset"));
  Object.assign($("k"), { min: k.min, max: k.max, value: url.get("k") ?? k.default });
  if (url.has("query")) $("query").value = url.get("query");

  $("dataset").addEventListener("change", () => onDatasetChange().catch((e) => setStatus(e.message, true)));
  $("measure").addEventListener("change", () => loadQueryIds().catch((e) => setStatus(e.message, true)));
  $("random").addEventListener("click", pickRandomQuery);
  $("search-form").addEventListener("submit", search);
  for (const id of ["show-both", "show-model", "show-truth"]) $(id).addEventListener("change", applyVisibility);

  try {
    await onDatasetChange({ model: url.get("model"), measure: url.get("measure") });
    if (url.has("query")) {
      $("query").value = url.get("query");
      await search();
    } else {
      setStatus("Pick a query trajectory and press Search.");
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

init();
