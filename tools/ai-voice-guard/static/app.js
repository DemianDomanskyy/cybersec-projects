"use strict";

const state = {
  models: [],
  defaultModel: null,
  selectedModels: [],
  history: JSON.parse(sessionStorage.getItem("vg_history") || "[]"),
  mediaRecorder: null,
  recordedChunks: [],
};

const $ = (sel) => document.querySelector(sel);

// FastAPI error bodies vary: a plain string detail, or (for request
// validation errors) a list of {loc, msg, type} objects. Handle both so a
// server-side error never renders as a bare "[object Object]".
function errorMessage(body, fallback) {
  if (!body || !body.detail) return fallback;
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return JSON.stringify(body.detail);
}

// ---------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------

async function checkAuth() {
  const res = await fetch("/api/me");
  if (res.ok) {
    await showApp();
  } else {
    showLogin();
  }
}

function showLogin() {
  $("#app-view").hidden = true;
  $("#login-view").hidden = false;
}

async function showApp() {
  $("#login-view").hidden = true;
  $("#app-view").hidden = false;
  await loadModels();
  renderHistory();
}

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = $("#login-username").value;
  const password = $("#login-password").value;
  const errBox = $("#login-error");
  errBox.hidden = true;
  $("#login-submit").disabled = true;
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      errBox.textContent = errorMessage(body, "Wrong username or password.");
      errBox.hidden = false;
      return;
    }
    $("#login-password").value = "";
    await showApp();
  } finally {
    $("#login-submit").disabled = false;
  }
});

$("#logout-btn").addEventListener("click", async () => {
  await fetch("/api/logout", { method: "POST" });
  showLogin();
});

// ---------------------------------------------------------------------
// Model picker
// ---------------------------------------------------------------------

async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  state.models = data.models;
  state.defaultModel = data.default;
  state.selectedModels = [data.default];
  renderModelPicker();
}

function renderModelPicker() {
  const container = $("#model-picker");
  container.innerHTML = "";
  for (const name of state.models) {
    const checked = state.selectedModels.includes(name);
    const label = document.createElement("label");
    label.className = "model-option" + (checked ? " checked" : "");
    label.innerHTML = `<input type="checkbox" value="${name}" ${checked ? "checked" : ""}> <span>${name}</span>`;
    label.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) {
        if (!state.selectedModels.includes(name)) state.selectedModels.push(name);
      } else {
        state.selectedModels = state.selectedModels.filter((m) => m !== name);
      }
      renderModelPicker();
      scheduleAnalysis();
    });
    container.appendChild(label);
  }
  $("#active-model-count").textContent = state.selectedModels.length;
}

// ---------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------

const lastAudio = { blob: null, filename: null };

$("#dropzone").addEventListener("click", () => $("#file-input").click());
$("#dropzone").addEventListener("dragover", (e) => {
  e.preventDefault();
  $("#dropzone").classList.add("dragover");
});
$("#dropzone").addEventListener("dragleave", () => $("#dropzone").classList.remove("dragover"));
$("#dropzone").addEventListener("drop", (e) => {
  e.preventDefault();
  $("#dropzone").classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], e.dataTransfer.files[0].name);
});
$("#file-input").addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0], e.target.files[0].name);
});

function handleFile(blob, filename) {
  lastAudio.blob = blob;
  lastAudio.filename = filename;
  const url = URL.createObjectURL(blob);
  const preview = $("#audio-preview");
  preview.src = url;
  preview.hidden = false;
  runAnalysis(blob, filename);
}

// ---------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------

$("#record-btn").addEventListener("click", async () => {
  const btn = $("#record-btn");
  const status = $("#record-status");

  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    state.mediaRecorder.stop();
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    status.textContent = "Microphone access denied or unavailable.";
    return;
  }

  state.recordedChunks = [];
  const recorder = new MediaRecorder(stream);
  state.mediaRecorder = recorder;

  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) state.recordedChunks.push(e.data);
  };
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    btn.textContent = "Start recording";
    btn.classList.remove("recording");
    status.textContent = "";
    const blob = new Blob(state.recordedChunks, { type: "audio/webm" });
    handleFile(blob, "mic-recording.webm");
  };

  recorder.start();
  btn.textContent = "Stop recording";
  btn.classList.add("recording");
  status.textContent = "Recording…";
});

// ---------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------

// Toggling several model checkboxes quickly (or re-uploading fast) can
// fire overlapping /api/analyze requests. Without this guard, whichever
// slow request happened to resolve last would win and overwrite a newer
// result on screen — and every stale request left running would keep
// burning real server-side inference time for a result that's about to be
// thrown away. abortController cancels the previous in-flight request
// outright; the generation counter is a second layer that drops a stale
// response even if it slips in before the abort takes effect.
let analysisAbortController = null;
let analysisGeneration = 0;
let scheduleTimer = null;

// Debounce rapid-fire model checkbox changes into one request after things
// settle, instead of firing (and immediately aborting) one per click.
function scheduleAnalysis() {
  if (!lastAudio.blob) return;
  clearTimeout(scheduleTimer);
  scheduleTimer = setTimeout(() => runAnalysis(lastAudio.blob, lastAudio.filename), 400);
}

async function runAnalysis(blob, filename) {
  if (state.selectedModels.length === 0) {
    $("#results").innerHTML = `<p class="dim">Pick at least one model in the sidebar to run an analysis.</p>`;
    return;
  }

  if (analysisAbortController) analysisAbortController.abort();
  const controller = new AbortController();
  analysisAbortController = controller;
  const myGeneration = ++analysisGeneration;

  const results = $("#results");
  results.innerHTML = `<div class="spinner-row"><div class="spinner"></div><span>Analyzing…</span></div>`;

  const form = new FormData();
  form.append("file", blob, filename);
  form.append("models", JSON.stringify(state.selectedModels));

  const start = performance.now();
  let res;
  try {
    res = await fetch("/api/analyze", { method: "POST", body: form, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") return; // superseded by a newer request
    if (myGeneration === analysisGeneration) {
      results.innerHTML = `<div class="alert alert-error">Request failed: ${escapeHtml(String(err))}</div>`;
    }
    return;
  }
  const elapsed = ((performance.now() - start) / 1000).toFixed(2);
  if (myGeneration !== analysisGeneration) return; // a newer request has already started

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    results.innerHTML = `<div class="alert alert-error">${escapeHtml(errorMessage(body, "Analysis failed."))}</div>`;
    return;
  }

  const analysis = await res.json();
  if (myGeneration !== analysisGeneration) return;
  renderResults(analysis, filename, elapsed);
  addHistory(filename, analysis);
}

function suspicionColor(fakeProb) {
  // Warm sunset (Tangerine Dream) sliding through Blush Rose into
  // Grape Soda as suspicion climbs, matching the app's palette.
  const stops = [
    [0xfc, 0xa1, 0x7d], // accent — Tangerine Dream
    [0xda, 0x62, 0x7d], // danger — Blush Rose
    [0x9a, 0x34, 0x8e], // danger-dim — Grape Soda
  ];
  const t = Math.max(0, Math.min(1, fakeProb));
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const localT = scaled - i;
  const rgb = stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * localT));
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

function renderResults(analysis, sourceLabel, elapsed) {
  const results = analysis.results;
  const container = $("#results");
  container.innerHTML = "";

  if (results.length === 1) {
    container.appendChild(buildSingleVerdict(results[0], sourceLabel, analysis.duration));
  } else {
    container.appendChild(buildConsensus(results));
    container.appendChild(buildModelGrid(results));
  }

  const chartRow = document.createElement("div");
  chartRow.className = "chart-row chart-section";

  const waveDiv = document.createElement("div");
  waveDiv.className = "chart-box";
  const waveCaption = document.createElement("div");
  waveCaption.className = "chart-caption";
  waveCaption.textContent = results.length === 1 ? "" : "WAVEFORM";
  const wavePlot = document.createElement("div");
  waveDiv.appendChild(waveCaption);
  waveDiv.appendChild(wavePlot);

  if (results.length === 1) {
    const gaugeDiv = document.createElement("div");
    gaugeDiv.className = "chart-box";
    const gaugeCaption = document.createElement("div");
    gaugeCaption.className = "chart-caption";
    gaugeCaption.textContent = "FAKE-VOICE PROBABILITY";
    const gaugePlot = document.createElement("div");
    gaugeDiv.appendChild(gaugeCaption);
    gaugeDiv.appendChild(gaugePlot);
    chartRow.appendChild(gaugeDiv);
    waveCaption.textContent = "WAVEFORM";
    plotGauge(gaugePlot, results[0].fake_prob);
  }

  chartRow.appendChild(waveDiv);
  container.appendChild(chartRow);
  plotWaveform(wavePlot, analysis.waveform, results[0].is_fake);

  if (analysis.timeline && analysis.timeline.length) {
    const tCaption = document.createElement("div");
    tCaption.className = "chart-caption chart-section";
    tCaption.textContent = "SUSPICION TIMELINE";
    const tBox = document.createElement("div");
    tBox.className = "chart-box";
    const tPlot = document.createElement("div");
    const tNote = document.createElement("p");
    tNote.className = "dim small";
    tNote.textContent = `Each bar is an independent 2s-window score from ${analysis.timeline_model} — spikes show where the clip itself sounds most synthetic, rather than averaging that away into one number.`;
    tBox.appendChild(tPlot);
    container.appendChild(tCaption);
    container.appendChild(tBox);
    container.appendChild(tNote);
    plotTimeline(tPlot, analysis.timeline, analysis.duration);
  }

  const timeCaption = document.createElement("p");
  timeCaption.className = "dim small";
  timeCaption.textContent = `${elapsed}s`;
  container.appendChild(timeCaption);

  const downloadBtn = document.createElement("button");
  downloadBtn.className = "btn";
  downloadBtn.textContent = "Download report (JSON)";
  downloadBtn.addEventListener("click", () => downloadReport(analysis, sourceLabel));
  container.appendChild(downloadBtn);
}

function buildSingleVerdict(result, sourceLabel, duration) {
  const div = document.createElement("div");
  const cls = result.is_fake ? "verdict-fake" : "verdict-real";
  const label = result.is_fake ? "SOUNDS AI-GENERATED" : "SOUNDS HUMAN";
  const icon = result.is_fake ? "⚠️" : "✅";
  div.className = `verdict-card ${cls}`;
  div.innerHTML = `
    <div class="verdict-label">${icon} ${label}</div>
    <div class="verdict-sub">${(result.confidence * 100).toFixed(1)}% confidence · ${duration.toFixed(1)}s clip · ${escapeHtml(sourceLabel)}</div>
  `;
  return div;
}

function buildConsensus(results) {
  const fakeVotes = results.filter((r) => r.is_fake).length;
  const total = results.length;
  const avgFakeProb = results.reduce((s, r) => s + r.fake_prob, 0) / total;
  const majorityFake = fakeVotes > total / 2;
  const label = majorityFake ? "AI-generated" : "Human";
  const icon = majorityFake ? "⚠️" : "✅";
  const div = document.createElement("div");
  div.className = "consensus-banner";
  div.innerHTML = `${icon} <b>${fakeVotes} of ${total} models</b> say <b>${label}</b> · average fake-voice probability ${(avgFakeProb * 100).toFixed(1)}%`;
  return div;
}

function buildModelGrid(results) {
  const grid = document.createElement("div");
  grid.className = "model-grid";
  for (const r of results) {
    const card = document.createElement("div");
    card.className = "model-card";
    const tag = r.is_fake ? "fake" : "real";
    const color = r.is_fake ? "#DA627D" : "#FCA17D";
    const verdictText = r.is_fake ? "AI-generated" : "Human";
    card.innerHTML = `
      <div class="model-card-name">${escapeHtml(r.model_name)}</div>
      <div class="model-card-verdict ${tag}">${verdictText}</div>
      <div class="dim small">${(r.confidence * 100).toFixed(1)}% confidence</div>
      <div class="model-card-bar-bg"><div class="model-card-bar-fill" style="width:${(r.fake_prob * 100).toFixed(1)}%; background:${color};"></div></div>
    `;
    grid.appendChild(card);
  }
  return grid;
}

function plotGauge(el, fakeProb) {
  const color = fakeProb > 0.5 ? "#DA627D" : "#FCA17D";
  Plotly.newPlot(el, [{
    type: "indicator",
    mode: "gauge+number",
    value: fakeProb * 100,
    number: { suffix: "%", font: { color, size: 34 } },
    gauge: {
      axis: { range: [0, 100], tickcolor: "#B79A8C", tickfont: { color: "#B79A8C" } },
      bar: { color },
      bgcolor: "#1B0E33",
      borderwidth: 0,
      steps: [
        { range: [0, 50], color: "rgba(252,161,125,0.12)" },
        { range: [50, 100], color: "rgba(218,98,125,0.12)" },
      ],
      threshold: { line: { color: "#F9DBBD", width: 2 }, thickness: 0.8, value: 50 },
    },
  }], {
    height: 200,
    margin: { l: 20, r: 20, t: 30, b: 10 },
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#B79A8C" },
  }, { displayModeBar: false, responsive: true });
}

function plotWaveform(el, waveform, isFake) {
  const color = isFake ? "#DA627D" : "#FCA17D";
  Plotly.newPlot(el, [{
    type: "scatter",
    x: waveform.x,
    y: waveform.y,
    line: { color, width: 1 },
    fill: "tozeroy",
    fillcolor: isFake ? "rgba(218,98,125,0.18)" : "rgba(252,161,125,0.18)",
  }], {
    height: 160,
    margin: { l: 0, r: 0, t: 10, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { showgrid: false, color: "#B79A8C", title: "seconds" },
    yaxis: { showgrid: false, color: "#B79A8C", showticklabels: false },
    showlegend: false,
  }, { displayModeBar: false, responsive: true });
}

function plotTimeline(el, timeline, duration) {
  const times = timeline.map((p) => p.time);
  const probs = timeline.map((p) => p.fake_prob * 100);
  const colors = timeline.map((p) => suspicionColor(p.fake_prob));
  const width = times.length > 1 ? (times[1] - times[0]) * 0.85 : 1.0;

  Plotly.newPlot(el, [{
    type: "bar",
    x: times,
    y: probs,
    width,
    marker: { color: colors, line: { width: 0 }, cornerradius: 6 },
    hovertemplate: "%{x:.1f}s · %{y:.0f}%% fake<extra></extra>",
  }], {
    height: 140,
    margin: { l: 0, r: 0, t: 10, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { showgrid: false, color: "#B79A8C", title: "seconds", range: [0, duration] },
    yaxis: { showgrid: false, color: "#B79A8C", range: [0, 100], title: "% fake" },
    shapes: [{ type: "line", x0: 0, x1: duration, y0: 50, y1: 50, line: { color: "#B79A8C", width: 1, dash: "dot" } }],
    showlegend: false,
    bargap: 0.2,
  }, { displayModeBar: false, responsive: true });
}

function downloadReport(analysis, sourceLabel) {
  const report = {
    source: sourceLabel,
    duration_seconds: Math.round(analysis.duration * 100) / 100,
    analyzed_at: new Date().toISOString(),
    results: analysis.results.map((r) => ({
      model: r.model_name,
      model_id: r.model_id,
      verdict: r.is_fake ? "ai-generated" : "human",
      confidence: Math.round(r.confidence * 10000) / 10000,
      fake_probability: Math.round(r.fake_prob * 10000) / 10000,
    })),
    suspicion_timeline: (analysis.timeline || []).map((p) => ({
      time_seconds: Math.round(p.time * 100) / 100,
      fake_probability: Math.round(p.fake_prob * 10000) / 10000,
    })),
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  a.href = url;
  a.download = `voice-guard-report-${stamp}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------
// History
// ---------------------------------------------------------------------

function addHistory(sourceLabel, analysis) {
  const results = analysis.results;
  const fakeVotes = results.filter((r) => r.is_fake).length;
  const isFake = fakeVotes > results.length / 2;
  const avgConfidence = results.reduce((s, r) => s + r.confidence, 0) / results.length;
  state.history.unshift({
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    source: sourceLabel,
    isFake,
    confidence: avgConfidence,
    models: results.length,
  });
  state.history = state.history.slice(0, 20);
  sessionStorage.setItem("vg_history", JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  const list = $("#history-list");
  if (state.history.length === 0) {
    list.innerHTML = `<p class="dim">No scans yet this session.</p>`;
    return;
  }
  list.innerHTML = state.history.map((h) => {
    const tagClass = h.isFake ? "tag-fake" : "tag-real";
    const tagText = h.isFake ? "AI" : "HUMAN";
    const modelsNote = h.models > 1 ? ` · ${h.models} models` : "";
    return `<div class="history-row">
      <span>${h.time} · ${escapeHtml(h.source)}${modelsNote}</span>
      <span class="${tagClass}">${tagText} ${(h.confidence * 100).toFixed(0)}%</span>
    </div>`;
  }).join("");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

checkAuth();
