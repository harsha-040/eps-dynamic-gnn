const API = "";
const FEATURE_COLORS = {
  voltage: "#6d93ff", current: "#ffbf5c", power: "#35d6a8",
  temperature: "#ff7676", soc: "#b98cf0",
};
const FEATURE_UNITS = { voltage: "V", current: "A", power: "W", temperature: "°C", soc: "" };
const NODE_POS = { SA: [70, 90], PCU: [245, 90], BAT: [420, 90], BUS: [245, 250], LOAD: [420, 250] };
const MODEL_COLORS = { cnn_baseline: "#6d93ff", lstm_baseline: "#ffbf5c", static_gnn: "#ff7676", dynamic_gnn: "#35d6a8" };
const MODEL_NAMES = { cnn_baseline: "CNN", lstm_baseline: "LSTM", static_gnn: "Static-GNN", dynamic_gnn: "Dynamic-GNN" };

let state = { meta: null, run: null, playing: false, playTimer: null };

async function getJSON(url, opts) {
  const res = await fetch(API + url, opts);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = "status" + (cls ? " " + cls : "");
}

function makeTooltip(container) {
  const tip = document.createElement("div");
  tip.className = "chart-tooltip";
  container.appendChild(tip);
  return tip;
}

// ---------------------------------------------------------------- init ----
async function init() {
  state.meta = await getJSON("/api/meta");
  const sel = document.getElementById("faultSelect");
  sel.innerHTML = state.meta.fault_classes.map(c => `<option value="${c}">${c}</option>`).join("");

  document.getElementById("runBtn").addEventListener("click", runSimulation);
  document.getElementById("timeSlider").addEventListener("input", onSliderChange);
  document.getElementById("playBtn").addEventListener("click", togglePlay);

  await loadPerformanceTable();
  await loadHistoryChart();
  await runSimulation();
}

// ------------------------------------------------------------ run sim -----
async function runSimulation() {
  setStatus("running simulator + 4 models…");
  document.getElementById("runBtn").disabled = true;
  try {
    const fault_type = document.getElementById("faultSelect").value;
    const seedVal = document.getElementById("seedInput").value;
    const body = { fault_type };
    if (seedVal !== "") body.seed = parseInt(seedVal, 10);
    const t0 = performance.now();
    const run = await getJSON("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    state.run = run;
    const dt = (performance.now() - t0).toFixed(0);
    setStatus(`done in ${dt} ms — ${run.fault_event.mode} fault, onset ${(run.fault_event.onset_sec/60).toFixed(1)} min, duration ${(run.fault_event.duration_sec/60).toFixed(1)} min`, "ok");

    renderTelemetry(run);
    renderPredictions(run);
    if (run.dynamic_graph) {
      const slider = document.getElementById("timeSlider");
      slider.max = run.t_minutes.length - 1;
      slider.value = 0;
      renderGraphAtStep(0);
    }
  } catch (e) {
    console.error(e);
    setStatus("error: " + e.message, "err");
  } finally {
    document.getElementById("runBtn").disabled = false;
  }
}

// --------------------------------------------------------- telemetry ------
function renderTelemetry(run) {
  const grid = document.getElementById("telemetryGrid");
  grid.innerHTML = "";
  for (const node of state.meta.node_names) {
    const card = document.createElement("div");
    card.className = "node-card";
    const canvasId = `chart_${node}`;
    card.innerHTML = `<h3>${node}</h3>
      <div class="chart-wrap"><canvas id="${canvasId}" width="520" height="160" style="width:100%;height:160px"></canvas></div>
      <div class="legend" id="legend_${node}"></div>`;
    grid.appendChild(card);
    drawTelemetryChart(canvasId, `legend_${node}`, run, node);
  }
}

function drawTelemetryChart(canvasId, legendId, run, node) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const padTop = 8, padBottom = 20, padL = 4, padR = 4;
  const plotH = H - padTop - padBottom;

  const t = run.t_minutes;
  const n = t.length;
  const tMin = t[0], tMax = t[t.length - 1];
  const xOf = (i) => padL + ((t[i] - tMin) / (tMax - tMin)) * (W - padL - padR);
  const tooltip = makeTooltip(canvas.parentElement);

  // build the list of series actually drawn (skip flat/not-applicable channels)
  const feats = state.meta.feature_names;
  const seriesInfo = [];
  for (const feat of feats) {
    const series = run.telemetry[node][feat];
    const lo = Math.min(...series), hi = Math.max(...series);
    if (hi - lo < 1e-6) continue;
    seriesInfo.push({ feat, series, lo, hi, yOf: (v) => padTop + plotH - ((v - lo) / (hi - lo)) * plotH });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // gridlines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = padTop + (plotH / 4) * g;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    }

    // shading: eclipse (gray) and fault-active (red)
    const shade = (mask, color) => {
      ctx.fillStyle = color;
      let start = null;
      for (let i = 0; i <= n; i++) {
        const on = i < n && mask[i];
        if (on && start === null) start = i;
        if (!on && start !== null) {
          ctx.fillRect(xOf(start), padTop, xOf(i - 1) - xOf(start) + 2, plotH);
          start = null;
        }
      }
    };
    shade(run.eclipse_mask, "rgba(255,255,255,0.05)");
    shade(run.fault_mask, "rgba(255,118,118,0.10)");

    for (const s of seriesInfo) {
      ctx.strokeStyle = FEATURE_COLORS[s.feat];
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = xOf(i), y = s.yOf(s.series[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // x-axis time labels
    ctx.fillStyle = "#6b7695";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("0 min", padL, H - 6);
    ctx.textAlign = "right";
    ctx.fillText(`${tMax.toFixed(0)} min`, W - padR, H - 6);
  }
  draw();

  const legend = document.getElementById(legendId);
  legend.innerHTML = seriesInfo.map(s =>
    `<span><span class="swatch" style="background:${FEATURE_COLORS[s.feat]}"></span>${s.feat}${FEATURE_UNITS[s.feat] ? ` (${FEATURE_UNITS[s.feat]})` : ""}</span>`
  ).join("");

  // ---- hover tooltip ----
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = W / rect.width;
    const mx = (e.clientX - rect.left) * scaleX;
    let idx = Math.round(((mx - padL) / (W - padL - padR)) * (n - 1));
    idx = Math.max(0, Math.min(n - 1, idx));
    const eclipse = run.eclipse_mask[idx], fault = run.fault_mask[idx];
    let html = `<b>t = ${t[idx].toFixed(1)} min</b> ${eclipse ? "🌑" : "☀️"}${fault ? " ⚠️" : ""}<br>`;
    for (const s of seriesInfo) {
      html += `<span style="color:${FEATURE_COLORS[s.feat]}">${s.feat}</span>: ${s.series[idx].toFixed(2)}${FEATURE_UNITS[s.feat]}<br>`;
    }
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
    const leftPct = (e.clientX - rect.left) / rect.width;
    tooltip.style.left = leftPct > 0.65 ? "auto" : `${e.clientX - rect.left + 12}px`;
    tooltip.style.right = leftPct > 0.65 ? `${rect.width - (e.clientX - rect.left) + 12}px` : "auto";
    tooltip.style.top = `${e.clientY - rect.top - 10}px`;
  });
  canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
}

// -------------------------------------------------------- predictions -----
function renderPredictions(run) {
  const grid = document.getElementById("predictionsGrid");
  grid.innerHTML = "";
  const order = ["cnn_baseline", "lstm_baseline", "static_gnn", "dynamic_gnn"];
  for (const key of order) {
    const p = run.predictions[key];
    if (!p) continue;
    const correct = p.pred_label === run.true_label;
    const card = document.createElement("div");
    card.className = "model-card";
    let rows = "";
    for (const c of state.meta.fault_classes) {
      const prob = p.probs[c];
      const isTrue = c === run.true_label;
      rows += `<div class="prob-row">
        <span class="label${isTrue ? " is-true" : ""}">${c}${isTrue ? " ✓ true" : ""}</span>
        <span class="prob-bar-bg"><span class="prob-bar${isTrue ? " true-label" : ""}" style="width:${(prob*100).toFixed(1)}%"></span></span>
        <span class="prob-val">${(prob*100).toFixed(1)}%</span>
      </div>`;
    }
    card.innerHTML = `<h3>${p.display_name} <span class="badge ${correct ? "correct" : "wrong"}">${correct ? "✓ correct" : "✗ wrong"} — guessed ${p.pred_label}</span></h3>${rows}`;
    grid.appendChild(card);
  }
}

// --------------------------------------------------------- perf table -----
async function loadPerformanceTable() {
  try {
    const data = await getJSON("/api/performance");
    const best = Math.max(...data.results.map(r => r.accuracy));
    const tbody = document.querySelector("#perfTable tbody");
    tbody.innerHTML = data.results.map(r => `
      <tr class="${r.accuracy === best ? "best" : ""}">
        <td>${r.display_name}</td>
        <td>${r.n_params.toLocaleString()}</td>
        <td>${(r.accuracy*100).toFixed(1)}%</td>
        <td>${(r.recall_macro*100).toFixed(1)}%</td>
        <td>${(r.f1_macro*100).toFixed(1)}%</td>
        <td>${r.auc_macro_ovr.toFixed(3)}</td>
      </tr>`).join("");
  } catch (e) {
    console.warn("no performance.json yet", e);
  }
}

async function loadHistoryChart() {
  try {
    const all = await getJSON("/api/history_all");
    const canvas = document.getElementById("historyChart");
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const padL = 34, padR = 10, padTop = 10, padBottom = 22;
    const plotW = W - padL - padR, plotH = H - padTop - padBottom;
    const tooltip = makeTooltip(canvas.parentElement);

    const series = {};
    let maxEpochs = 0;
    for (const [name, data] of Object.entries(all)) {
      series[name] = data.history.map(h => h.val_acc);
      maxEpochs = Math.max(maxEpochs, series[name].length);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      // gridlines + y-axis labels (0-100%)
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.fillStyle = "#6b7695";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "right";
      for (let g = 0; g <= 4; g++) {
        const frac = g / 4;
        const y = padTop + plotH - frac * plotH;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillText(`${(frac * 100).toFixed(0)}%`, padL - 6, y + 3);
      }
      ctx.textAlign = "left";
      ctx.fillText("epoch 0", padL, H - 6);
      ctx.textAlign = "right";
      ctx.fillText(`epoch ${maxEpochs}`, W - padR, H - 6);

      let legendY = padTop + 4;
      for (const [name, vals] of Object.entries(series)) {
        const n = vals.length;
        ctx.strokeStyle = MODEL_COLORS[name] || "#888";
        ctx.lineWidth = 2;
        ctx.beginPath();
        vals.forEach((v, i) => {
          const x = padL + (i / (n - 1)) * plotW;
          const y = padTop + plotH - v * plotH;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = MODEL_COLORS[name] || "#888";
        ctx.fillRect(W - 118, legendY, 8, 8);
        ctx.fillStyle = "#8d99bd";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`${MODEL_NAMES[name] || name} (${(vals[vals.length-1]*100).toFixed(0)}%)`, W - 106, legendY + 8);
        legendY += 15;
      }
    }
    draw();

    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = W / rect.width;
      const mx = (e.clientX - rect.left) * scaleX;
      const frac = Math.max(0, Math.min(1, (mx - padL) / plotW));
      let html = `<b>epoch ${Math.round(frac * maxEpochs)}</b><br>`;
      for (const [name, vals] of Object.entries(series)) {
        const idx = Math.min(vals.length - 1, Math.round(frac * (vals.length - 1)));
        html += `<span style="color:${MODEL_COLORS[name]}">${MODEL_NAMES[name] || name}</span>: ${(vals[idx]*100).toFixed(1)}%<br>`;
      }
      tooltip.innerHTML = html;
      tooltip.style.display = "block";
      tooltip.style.left = `${e.clientX - rect.left + 12}px`;
      tooltip.style.top = `${e.clientY - rect.top - 10}px`;
    });
    canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
  } catch (e) {
    console.warn("no training history yet", e);
  }
}

// -------------------------------------------------------- dynamic graph ---
function onSliderChange(e) {
  renderGraphAtStep(parseInt(e.target.value, 10));
}

function togglePlay() {
  const btn = document.getElementById("playBtn");
  state.playing = !state.playing;
  btn.textContent = state.playing ? "⏸ Pause" : "▶ Play";
  if (state.playing) {
    state.playTimer = setInterval(() => {
      const slider = document.getElementById("timeSlider");
      let v = parseInt(slider.value, 10) + 2;
      if (v > parseInt(slider.max, 10)) v = 0;
      slider.value = v;
      renderGraphAtStep(v);
    }, 80);
  } else {
    clearInterval(state.playTimer);
  }
}

function renderGraphAtStep(step) {
  const run = state.run;
  if (!run || !run.dynamic_graph) return;
  const A = run.dynamic_graph.a_soft[step]; // (N,N), A[i][j] = weight j->i
  const nodes = state.meta.node_names;
  const tMin = run.t_minutes[step];
  const eclipse = run.eclipse_mask[step];
  const fault = run.fault_mask[step];
  document.getElementById("timeLabel").innerHTML =
    `t = ${tMin.toFixed(1)} min &nbsp; ` +
    `<span class="state-pill ${eclipse ? "eclipse" : "sunlit"}">${eclipse ? "🌑 eclipse" : "☀️ sunlit"}</span>` +
    (fault ? ` <span class="state-pill fault">⚠ fault-active</span>` : "");

  drawGraphDiagram(A, nodes);
  drawHeatmap(A, nodes);
}

function drawGraphDiagram(A, nodes) {
  const svg = document.getElementById("graphSvg");
  svg.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";

  // top-3 sources per destination (mirrors the model's own GAT sparsification)
  const maxW = Math.max(...A.flat());
  const edges = [];
  for (let i = 0; i < nodes.length; i++) {
    const row = A[i].map((w, j) => [w, j]).filter(([, j]) => j !== i);
    row.sort((a, b) => b[0] - a[0]);
    row.slice(0, 3).forEach(([w, j]) => edges.push([j, i, w]));
  }

  const defs = document.createElementNS(ns, "defs");
  defs.innerHTML = `<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#6d93ff"></path></marker>`;
  svg.appendChild(defs);

  for (const [j, i, w] of edges) {
    const [x1, y1] = NODE_POS[nodes[j]];
    const [x2, y2] = NODE_POS[nodes[i]];
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const shrink = 30;
    const x1s = x1 + (dx / len) * shrink, y1s = y1 + (dy / len) * shrink;
    const x2s = x2 - (dx / len) * shrink, y2s = y2 - (dy / len) * shrink;
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", x1s); line.setAttribute("y1", y1s);
    line.setAttribute("x2", x2s); line.setAttribute("y2", y2s);
    line.setAttribute("stroke", "#6d93ff");
    line.setAttribute("stroke-width", (0.6 + (w / maxW) * 5).toFixed(2));
    line.setAttribute("stroke-opacity", (0.15 + 0.75 * (w / maxW)).toFixed(2));
    line.setAttribute("marker-end", "url(#arrow)");
    const title = document.createElementNS(ns, "title");
    title.textContent = `${nodes[j]} → ${nodes[i]}: weight ${w.toFixed(3)}`;
    line.appendChild(title);
    svg.appendChild(line);
  }

  for (const name of nodes) {
    const [x, y] = NODE_POS[name];
    const g = document.createElementNS(ns, "g");
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", x); circle.setAttribute("cy", y); circle.setAttribute("r", 26);
    circle.setAttribute("fill", "#1a2338");
    circle.setAttribute("stroke", "#35d6a8");
    circle.setAttribute("stroke-width", 1.5);
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", x); text.setAttribute("y", y + 4);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#eaeefb");
    text.setAttribute("font-size", "13");
    text.setAttribute("font-weight", "700");
    text.textContent = name;
    g.appendChild(circle); g.appendChild(text);
    svg.appendChild(g);
  }
}

function drawHeatmap(A, nodes) {
  const svg = document.getElementById("heatmapSvg");
  svg.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";
  const n = nodes.length;
  const padL = 60, padTop = 34, size = 260;
  const cell = size / n;
  const maxW = Math.max(...A.flat());

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const w = A[i][j];
      const t = maxW > 0 ? w / maxW : 0;
      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", padL + j * cell);
      rect.setAttribute("y", padTop + i * cell);
      rect.setAttribute("width", cell - 1.5);
      rect.setAttribute("height", cell - 1.5);
      rect.setAttribute("rx", 2);
      rect.setAttribute("fill", `rgba(109,147,255,${0.08 + 0.85 * t})`);
      const title = document.createElementNS(ns, "title");
      title.textContent = i === j ? `${nodes[i]} (self)` : `${nodes[j]} → ${nodes[i]}: ${w.toFixed(3)}`;
      rect.appendChild(title);
      svg.appendChild(rect);
      if (i !== j && t > 0.05) {
        const text = document.createElementNS(ns, "text");
        text.setAttribute("x", padL + j * cell + cell / 2);
        text.setAttribute("y", padTop + i * cell + cell / 2 + 4);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", "10");
        text.setAttribute("fill", t > 0.5 ? "#0a0e18" : "#8d99bd");
        text.textContent = w.toFixed(2);
        svg.appendChild(text);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const rowLabel = document.createElementNS(ns, "text");
    rowLabel.setAttribute("x", padL - 6);
    rowLabel.setAttribute("y", padTop + i * cell + cell / 2 + 4);
    rowLabel.setAttribute("text-anchor", "end");
    rowLabel.setAttribute("font-size", "11");
    rowLabel.setAttribute("fill", "#8d99bd");
    rowLabel.textContent = nodes[i];
    svg.appendChild(rowLabel);

    const colLabel = document.createElementNS(ns, "text");
    colLabel.setAttribute("x", padL + i * cell + cell / 2);
    colLabel.setAttribute("y", padTop - 10);
    colLabel.setAttribute("text-anchor", "middle");
    colLabel.setAttribute("font-size", "11");
    colLabel.setAttribute("fill", "#8d99bd");
    colLabel.textContent = nodes[i];
    svg.appendChild(colLabel);
  }
  const ylab = document.createElementNS(ns, "text");
  ylab.setAttribute("x", 14); ylab.setAttribute("y", padTop + size / 2);
  ylab.setAttribute("text-anchor", "middle"); ylab.setAttribute("font-size", "10"); ylab.setAttribute("fill", "#35d6a8");
  ylab.setAttribute("transform", `rotate(-90 14 ${padTop + size / 2})`);
  ylab.textContent = "← receiving node (row)";
  svg.appendChild(ylab);

  const xlab = document.createElementNS(ns, "text");
  xlab.setAttribute("x", padL + size / 2); xlab.setAttribute("y", 16);
  xlab.setAttribute("text-anchor", "middle"); xlab.setAttribute("font-size", "10"); xlab.setAttribute("fill", "#6d93ff");
  xlab.textContent = "sending node (column) →";
  svg.appendChild(xlab);
}

init();
