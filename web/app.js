/* Iceland thermal anomalies: time series of model residual statistics with per-image quicklooks.
   Data: index.json, monthly JSONL shards and product PNGs in the public bucket. */
(function () {
  "use strict";
  const CFG = window.THERMAL_CONFIG;
  const $ = (id) => document.getElementById(id);
  const PLATFORM_NAME = { MSG1: "Meteosat-8", MSG2: "Meteosat-9", MSG3: "Meteosat-10", MSG4: "Meteosat-11", MTI1: "MTG-I1" };
  const STAT_LABEL = {
    sum_pos: "Summed positive residual", max: "Maximum residual", max_z: "Maximum z-score",
    n_above: "Pixels above z = 4", vrp_w: "Radiative power (W)", mean: "Mean residual",
  };
  const IMAGES = [
    ["residual", "Residual (observed − predicted)"], ["zscore", "z-score"],
    ["ir39", "IR3.9 observed"], ["ir39_pred", "IR3.9 predicted"],
    ["ir108", "IR10.8"], ["vis06", "VIS0.6 (day only)"], ["clm", "Cloud mask"],
  ];
  const state = { index: null, rows: [], eruptions: [], selected: null, source: null, region: null,
                  stat: "sum_pos", range: "all" };
  let chart = null, cloudChart = null;

  // ---------- helpers ----------
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmtTime = (s) => new Date(s * 1000).toISOString().replace("T", " ").slice(0, 16) + " UTC";
  const fmtNum = (v, stat) => {
    if (v == null || Number.isNaN(v)) return "–";
    if (stat === "vrp_w") return v >= 1e9 ? (v / 1e9).toFixed(2) + " GW" : v >= 1e6 ? (v / 1e6).toFixed(1) + " MW" : Math.round(v) + " W";
    if (stat === "n_above") return String(v);
    return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3);
  };
  const stamp = (iso) => iso.replace(/[-:]/g, "").replace("Z", "");
  const productUrl = (row, name) => {
    const d = row.time.slice(0, 10).split("-");
    return `${CFG.dataBase}/products/${row.sat}/${d[0]}/${d[1]}/${d[2]}/${stamp(row.time)}/${name}.png`;
  };
  function setStatus(msg) { $("status").textContent = msg; }
  function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    const head = lines.shift().split(",");
    return lines.map((l) => {
      const cells = l.split(","); const o = {};
      head.forEach((h, i) => (o[h] = cells[i] || ""));
      return o;
    });
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    for (const k of ["source", "region", "stat", "range"]) if (p.get(k)) state[k] = p.get(k);
    if (p.get("t")) state.selected = p.get("t");
  }
  function writeHash() {
    const p = new URLSearchParams();
    p.set("source", state.source); p.set("region", state.region); p.set("stat", state.stat); p.set("range", state.range);
    if (state.selected) p.set("t", state.selected);
    history.replaceState(null, "", "#" + p.toString());
  }

  // ---------- data ----------
  async function loadIndex() {
    const r = await fetch(`${CFG.dataBase}/index.json`, { cache: "no-cache" });
    if (!r.ok) throw new Error("index.json " + r.status);
    state.index = await r.json();
    const sources = Object.keys(state.index.sources);
    fill($("source"), sources.map((s) => [s, s]), state.source || sources[0]);
    state.source = $("source").value;
    fillRegions();
  }
  function fillRegions() {
    const regions = Object.keys(state.index.sources[state.source].regions);
    fill($("region"), regions.map((r) => [r, r]), state.region || regions[0]);
    state.region = $("region").value;
  }
  function fill(sel, pairs, value) {
    sel.replaceChildren();
    for (const [v, label] of pairs) { const o = document.createElement("option"); o.value = v; o.textContent = label; sel.append(o); }
    if (pairs.some(([v]) => v === value)) sel.value = value;
  }
  async function loadRows() {
    const months = state.index.sources[state.source].regions[state.region].months;
    const wanted = monthsInRange(months);
    setStatus(`loading ${wanted.length} month${wanted.length === 1 ? "" : "s"}…`);
    const parts = await Promise.all(wanted.map(async (m) => {
      const r = await fetch(`${CFG.dataBase}/timeseries/${state.source}/${state.region}/${m}.jsonl`, { cache: "no-cache" });
      if (!r.ok) return [];
      return (await r.text()).split("\n").filter(Boolean).map((l) => JSON.parse(l));
    }));
    state.rows = parts.flat().sort((a, b) => a.time.localeCompare(b.time));
    const byPlatform = {};
    for (const r of state.rows) byPlatform[r.platform || "?"] = (byPlatform[r.platform || "?"] || 0) + 1;
    const platformParts = Object.entries(byPlatform).sort().map(([p, n]) => `${n} ${PLATFORM_NAME[p] || p}`);
    setStatus(`${state.rows.length} images (${platformParts.join(", ")})`);
  }
  // A range is "all", a number of days back from now, or a fixed window "YYYY-MM..YYYY-MM".
  function rangeWindow() {
    if (state.range === "all") return null;
    if (state.range.includes("..")) {
      const [a, b] = state.range.split("..");
      return { fromMonth: a, toMonth: b, from: Date.parse(a + "-01T00:00:00Z") / 1000,
               to: Date.parse(b + "-01T00:00:00Z") / 1000 + 32 * 86400 };
    }
    const from = Date.now() / 1000 - Number(state.range) * 86400;
    return { fromMonth: new Date(from * 1000).toISOString().slice(0, 7), toMonth: "9999-99", from, to: Infinity };
  }
  function monthsInRange(months) {
    const w = rangeWindow();
    if (!w) return months;
    return months.filter((m) => m >= w.fromMonth && m <= w.toMonth);
  }
  async function loadEruptions() {
    try {
      const r = await fetch(CFG.eruptionsCsv, { cache: "no-cache" });
      if (!r.ok) return;
      state.eruptions = parseCsv(await r.text()).filter((e) => e.start_date).map((e) => ({
        name: e.volcano, start: Date.parse(e.start_date + "T00:00:00Z") / 1000,
        end: e.end_date ? Date.parse(e.end_date + "T23:59:59Z") / 1000 : Date.now() / 1000,
      }));
    } catch (_) { /* optional */ }
  }

  // ---------- charts ----------
  function rowsInRange() {
    const w = rangeWindow();
    if (!w) return state.rows;
    return state.rows.filter((r) => { const t = Date.parse(r.time) / 1000; return t >= w.from && t < w.to; });
  }
  function bandsPlugin(getBands, color) {
    return { hooks: { drawClear: [(u) => {
      const ctx = u.ctx; ctx.save(); ctx.fillStyle = color;
      const top = u.bbox.top, h = u.bbox.height;
      for (const [a, b] of getBands(u)) {
        const x0 = Math.max(u.valToPos(a, "x", true), u.bbox.left);
        const x1 = Math.min(u.valToPos(b, "x", true), u.bbox.left + u.bbox.width);
        if (x1 > x0) ctx.fillRect(x0, top, x1 - x0, h);
      }
      ctx.restore();
    }] } };
  }
  const GAP_S = 3 * 3600; // a gap longer than this breaks lines and night bands
  function nightBands(rows) {
    // contiguous runs of samples with mean solar zenith > 100 deg, broken at data gaps
    const out = []; let start = null, last = null;
    for (const r of rows) {
      const t = Date.parse(r.time) / 1000, night = r.sza_mean != null && r.sza_mean > 100;
      const gap = last != null && t - last > GAP_S;
      if (start != null && (!night || gap)) { out.push([start, last]); start = null; }
      if (night && start == null) start = t;
      last = t;
    }
    if (start != null) out.push([start, last]);
    return out;
  }
  function withGaps(rows) {
    // insert a null sample inside every large gap so lines do not bridge missing data
    const out = [];
    for (let i = 0; i < rows.length; i++) {
      if (i > 0) {
        const a = Date.parse(rows[i - 1].time) / 1000, b = Date.parse(rows[i].time) / 1000;
        if (b - a > GAP_S) out.push(null);
      }
      out.push(rows[i]);
    }
    return out;
  }
  function eruptionBands() { return state.eruptions.map((e) => [e.start, e.end]); }

  function buildCharts() {
    const real = rowsInRange();
    const rows = withGaps(real); // null entries mark gaps; indices align with chart data
    let prevT = 0;
    const xs = rows.map((r) => { if (r) { prevT = Date.parse(r.time) / 1000; return prevT; } return prevT + 1; });
    const ys = rows.map((r) => (r && r[state.stat] != null ? r[state.stat] : null));
    const cloud = rows.map((r) => (r && r.n_valid ? (r.n_cloud || 0) / r.n_valid : null));
    const sat = rows.map((r, i) => (r && r.n_sat > 0 ? ys[i] : null));
    const nights = nightBands(real), erupt = eruptionBands();
    $("chart-title").textContent = STAT_LABEL[state.stat];
    if (chart) { chart.destroy(); chart = null; }
    if (cloudChart) { cloudChart.destroy(); cloudChart = null; }
    const series = css("--series"), muted = css("--muted"), grid = css("--grid"), axis = css("--axis");
    const axisOpts = { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { stroke: axis, width: 1 }, font: "12px system-ui, sans-serif" };
    const common = (el) => ({
      width: el.clientWidth, height: el.clientHeight,
      cursor: { sync: { key: "thermal" }, points: { size: 10, width: 2, stroke: series, fill: css("--surface") }, drag: { x: true, y: false } },
      legend: { show: false },
      scales: { x: { time: true } },
      axes: [{ ...axisOpts, space: 80 }, { ...axisOpts, size: 64, values: (u, vals) => vals.map((v) => fmtNum(v, state.stat)) }],
      plugins: [bandsPlugin(() => nights, css("--night")), bandsPlugin(() => erupt, css("--erupt"))],
    });
    chart = new uPlot({
      ...common($("chart")),
      series: [{}, { label: STAT_LABEL[state.stat], stroke: series, width: 2, spanGaps: false, points: { show: true, size: real.length < 400 ? 6 : 4, stroke: series, fill: css("--surface"), width: real.length < 400 ? 2 : 1.5 } },
                   { label: "saturated", stroke: css("--sat"), width: 0, points: { show: true, size: 12, stroke: css("--sat"), fill: css("--sat"), width: 1 }, paths: () => null }],
      hooks: {
        setCursor: [(u) => showTip(u, rows)],
        ready: [(u) => { u.over.addEventListener("click", () => { const i = u.cursor.idx; if (i != null && rows[i]) select(rows[i].time); }); }],
      },
    }, [xs, ys, sat], $("chart"));
    cloudChart = new uPlot({
      ...common($("cloud")),
      axes: [{ ...axisOpts, space: 80, show: false }, { ...axisOpts, size: 64, values: (u, vals) => vals.map((v) => Math.round(v * 100) + "%") }],
      scales: { x: { time: true }, y: { range: [0, 1] } },
      series: [{}, { label: "cloud fraction", stroke: css("--cloud"), width: 2, spanGaps: false }],
      hooks: { ready: [(u) => { u.over.addEventListener("click", () => { const i = u.cursor.idx; if (i != null && rows[i]) select(rows[i].time); }); }] },
    }, [xs, cloud], $("cloud"));
    buildTable(real);
    if (state.selected) showImages(state.selected);
  }
  function showTip(u, rows) {
    const tip = $("tip"), i = u.cursor.idx;
    if (i == null || !rows[i]) { tip.hidden = true; return; }
    const r = rows[i];
    tip.replaceChildren();
    const add = (v, l, strong) => { const d = document.createElement("div"); d.className = "row";
      const a = document.createElement("span"); a.className = strong ? "v" : ""; a.textContent = v;
      const b = document.createElement("span"); b.className = "l"; b.textContent = l; d.append(a, b); tip.append(d); };
    const head = document.createElement("div"); head.className = "l"; head.textContent = fmtTime(u.data[0][i]); tip.append(head);
    add(fmtNum(r[state.stat], state.stat), STAT_LABEL[state.stat], true);
    add(fmtNum(r.max_z, "max_z"), "max z");
    add(r.n_valid ? Math.round(100 * (r.n_cloud || 0) / r.n_valid) + "%" : "–", "cloud");
    if (r.n_sat) add(String(r.n_sat), "saturated px");
    add(r.sza_mean != null ? r.sza_mean.toFixed(0) + "°" : "–", "solar zenith");
    if (r.platform) add(PLATFORM_NAME[r.platform] || r.platform, "satellite");
    tip.hidden = false;
    const left = u.cursor.left, rect = u.over.getBoundingClientRect(), card = $("chart").closest(".card").getBoundingClientRect();
    const x = rect.left - card.left + left + 14, maxX = card.width - tip.offsetWidth - 8;
    tip.style.left = Math.min(x, maxX) + "px"; tip.style.top = (rect.top - card.top + 8) + "px";
  }
  function buildTable(rows) {
    const body = $("table").querySelector("tbody"); body.replaceChildren();
    for (const r of rows.slice().reverse()) {
      const tr = document.createElement("tr");
      const cells = [r.time.replace("T", " ").replace("Z", ""), fmtNum(r.sum_pos), fmtNum(r.max), fmtNum(r.max_z), String(r.n_above),
                     fmtNum(r.vrp_w, "vrp_w"), r.n_valid ? Math.round(100 * (r.n_cloud || 0) / r.n_valid) + "%" : "–", String(r.n_sat || 0), r.sza_mean != null ? r.sza_mean.toFixed(0) : "–"];
      for (const c of cells) { const td = document.createElement("td"); td.textContent = c; tr.append(td); }
      tr.addEventListener("click", () => select(r.time));
      body.append(tr);
    }
  }

  // ---------- images ----------
  function select(time) { state.selected = time; writeHash(); showImages(time); $("images").scrollIntoView({ behavior: "smooth", block: "nearest" }); }
  function showImages(time) {
    const rows = state.rows, i = rows.findIndex((r) => r.time === time);
    if (i < 0) return;
    const r = rows[i];
    $("images").hidden = false;
    $("img-title").textContent = `Images for ${time.replace("T", " ").replace("Z", " UTC")} (${r.platform || r.sat})`;
    $("img-stats").textContent = `${STAT_LABEL[state.stat]}: ${fmtNum(r[state.stat], state.stat)} · max z ${fmtNum(r.max_z, "max_z")} · ` +
      `cloud ${r.n_valid ? Math.round(100 * (r.n_cloud || 0) / r.n_valid) : "–"}% · saturated ${r.n_sat || 0} px · model ${r.model || "?"}`;
    const grid = $("img-grid"); grid.replaceChildren();
    for (const [name, label] of IMAGES) {
      if (name === "vis06" && r.sza_mean != null && r.sza_mean > 95) continue;
      const fig = document.createElement("figure"); const img = document.createElement("img");
      img.src = productUrl(r, name); img.alt = label; img.loading = "lazy";
      img.addEventListener("error", () => { fig.remove(); });
      const cap = document.createElement("figcaption"); cap.textContent = label; fig.append(img, cap); grid.append(fig);
    }
    $("prev").disabled = i === 0; $("next").disabled = i === rows.length - 1;
    $("prev").onclick = () => select(rows[i - 1].time); $("next").onclick = () => select(rows[i + 1].time);
  }

  // ---------- wiring ----------
  async function refresh() { await loadRows(); buildCharts(); writeHash(); }
  function wire() {
    $("source").addEventListener("change", async () => { state.source = $("source").value; fillRegions(); await refresh(); });
    $("region").addEventListener("change", async () => { state.region = $("region").value; await refresh(); });
    $("stat").addEventListener("change", () => { state.stat = $("stat").value; buildCharts(); writeHash(); });
    $("range").addEventListener("change", async () => { state.range = $("range").value; await refresh(); });
    $("theme").addEventListener("click", () => {
      const root = document.documentElement, dark = root.getAttribute("data-theme") === "dark" ||
        (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      root.setAttribute("data-theme", dark ? "light" : "dark");
      try { localStorage.setItem("theme", dark ? "light" : "dark"); } catch (_) {}
      buildCharts();
    });
    window.addEventListener("resize", () => {
      if (!chart) return;
      chart.setSize({ width: $("chart").clientWidth, height: $("chart").clientHeight });
      cloudChart.setSize({ width: $("cloud").clientWidth, height: $("cloud").clientHeight });
    });
    document.addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") $("prev").click(); if (e.key === "ArrowRight") $("next").click(); });
  }
  async function main() {
    try { const t = localStorage.getItem("theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (_) {}
    readHash();
    $("stat").value = state.stat; $("range").value = state.range;
    wire();
    try {
      await Promise.all([loadIndex(), loadEruptions()]);
      await refresh();
    } catch (err) { setStatus("could not load data: " + err.message); }
  }
  main();
})();
