// D-5: provider chooser + pipeline run + SSE streaming + step progress
// + source picker + queue + mass research + sub-theme cascade.

const $ = (id) => document.getElementById(id);

const STEPS = [
  { n: 1, title: "Index + theme" },
  { n: 2, title: "Search" },
  { n: 3, title: "Extract sources" },
  { n: 4, title: "Pass 1 — extract" },
  { n: 5, title: "Pass 2 — synthesize" },
  { n: 6, title: "Validate pack" },
  { n: 7, title: "Write pack" },
  { n: 8, title: "Wire topic tree" },
];

const state = {
  selectedProvider: null,
  providers: [],
  // active per-job SSE
  activeJobId: null,
  jobStream: null,
  // step state
  stepStatuses: STEPS.map(() => "pending"),
  currentStepIdx: -1,
  // source picker
  pickerResults: [],
  picking: false,
  // mode: "single" | "mass"
  mode: "single",
  // queue
  queueStream: null,
  queue: { pending: [], running: [], recent_done: [], concurrency: 1 },
  // cascade picker
  cascadeJobId: null,
  cascadeContext: null,
};

function setStatus(msg) {
  $("status").textContent = msg;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function setSelectedProviderLabel() {
  const text = state.selectedProvider
    ? `provider: ${state.selectedProvider}`
    : "no provider selected";
  $("selected-provider").textContent = text;
}

// ---------- providers ----------

function statusClass(p) {
  if (p.ready) return "ready";
  if (p.reachable) return "warn";
  return "bad";
}

function renderProviderCard(p) {
  const card = document.createElement("div");
  card.className = "provider-card";
  if (!p.ready) card.classList.add("unready");
  if (p.name === state.selectedProvider) card.classList.add("selected");

  const row1 = document.createElement("div");
  row1.className = "pc-row1";
  row1.innerHTML = `
    <span class="pc-name">${p.name}</span>
    <span class="pc-status ${statusClass(p)}">${p.status}</span>
  `;
  card.appendChild(row1);

  const row2 = document.createElement("div");
  if (p.error && !p.reachable) {
    row2.className = "pc-error";
    row2.textContent = p.error;
  } else {
    row2.className = "pc-row2";
    const lat = p.latency_ms ? ` · ${p.latency_ms}ms` : "";
    const models = p.available_models.length
      ? `${p.available_models.length} models`
      : "no models";
    row2.textContent = `${p.extract_model} / ${p.synthesize_model} · ${models}${lat}`;
  }
  card.appendChild(row2);

  if (p.ready) {
    card.addEventListener("click", () => selectProvider(p.name));
  }
  return card;
}

function renderProviderStrip() {
  const strip = $("provider-strip");
  strip.innerHTML = "";
  if (!state.providers.length) {
    strip.innerHTML = '<div class="loading">No providers configured.</div>';
    return;
  }
  for (const p of state.providers) {
    strip.appendChild(renderProviderCard(p));
  }
}

async function fetchState() {
  const r = await fetch("/api/state");
  const data = await r.json();
  state.selectedProvider = data.selected_provider;
  $("vault-path").textContent = `vault: ${data.vault_path}`;
  setSelectedProviderLabel();
  updateRunButton();
}

async function fetchProviders() {
  setStatus("probing providers…");
  const strip = $("provider-strip");
  strip.innerHTML = '<div class="loading">Probing providers…</div>';
  try {
    const r = await fetch("/api/providers");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    state.providers = await r.json();
    renderProviderStrip();
    setStatus(`probed ${state.providers.length} providers`);
  } catch (e) {
    strip.innerHTML = `<div class="loading">probe failed: ${e.message}</div>`;
    setStatus("probe failed");
  }
}

async function selectProvider(name) {
  setStatus(`selecting ${name}…`);
  try {
    const r = await fetch("/api/select-provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }
    const data = await r.json();
    state.selectedProvider = data.selected_provider;
    setSelectedProviderLabel();
    renderProviderStrip();
    setStatus(`selected ${name}`);
    updateRunButton();
  } catch (e) {
    setStatus(`select failed: ${e.message}`);
  }
}

// ---------- mode toggle ----------

function setMode(mode) {
  state.mode = mode;
  $("mode-single").classList.toggle("active", mode === "single");
  $("mode-mass").classList.toggle("active", mode === "mass");
  $("query-bar-single").hidden = mode !== "single";
  $("query-bar-mass").hidden   = mode !== "mass";
  updateRunButton();
  updateMassButton();
}

// ---------- run/queue gating ----------

function updateRunButton() {
  const hasQuery = $("query").value.trim().length > 0;
  const hasProvider = !!state.selectedProvider;
  const running = !!state.activeJobId;
  $("research-btn").disabled = !hasQuery || !hasProvider || running;
  $("cancel-btn").disabled = !running;
  if (!hasProvider) {
    $("research-btn").title = "Select a provider first";
  } else if (!hasQuery) {
    $("research-btn").title = "Enter a topic";
  } else if (running) {
    $("research-btn").title = "A job is already streaming";
  } else {
    $("research-btn").title = "";
  }
}

function massQueries() {
  return $("query-mass").value
    .split("\n")
    .map((q) => q.trim())
    .filter(Boolean);
}

function updateMassButton() {
  const qs = massQueries();
  const btn = $("queue-all-btn");
  btn.disabled = qs.length === 0 || !state.selectedProvider;
  btn.textContent = `Queue all (${qs.length})`;
}

// ---------- log ----------

function classifyLine(line) {
  if (line.startsWith("$ ")) return "log-cmd";
  if (/\b(error|failed|exception|traceback)\b/i.test(line)) return "log-bad";
  if (/\b(warn|warning)\b/i.test(line)) return "log-warn";
  if (line.startsWith("[runner]")) return "log-dim";
  if (/\b(done|wrote|saved|ok)\b/i.test(line)) return "log-ok";
  return "";
}

function appendLogLine(text) {
  const el = $("log");
  const span = document.createElement("span");
  const cls = classifyLine(text);
  if (cls) span.className = cls;
  span.textContent = text + "\n";
  el.appendChild(span);
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  if (nearBottom) el.scrollTop = el.scrollHeight;
}

function clearLog() {
  $("log").innerHTML = "";
  $("log-meta").textContent = "";
  $("log-title").textContent = "log";
}

// ---------- step progress ----------

function renderStepStrip() {
  const strip = $("step-strip");
  strip.innerHTML = "";
  STEPS.forEach((s, i) => {
    const status = state.stepStatuses[i];
    const li = document.createElement("li");
    li.className = `step ${status}`;
    let icon = "○";
    if (status === "running") icon = "⟳";
    else if (status === "done") icon = "✓";
    else if (status === "failed") icon = "✗";
    li.innerHTML = `
      <span class="step-icon">${icon}</span>
      <span class="step-num">${s.n}.</span>
      <span class="step-title">${s.title}</span>
    `;
    strip.appendChild(li);
  });
}

function resetSteps() {
  state.stepStatuses = STEPS.map(() => "pending");
  state.currentStepIdx = -1;
  renderStepStrip();
}

function advanceToStep(stepNum) {
  const idx = stepNum - 1;
  if (idx < 0 || idx >= STEPS.length) return;
  for (let i = 0; i < idx; i++) {
    if (state.stepStatuses[i] !== "failed") state.stepStatuses[i] = "done";
  }
  state.stepStatuses[idx] = "running";
  state.currentStepIdx = idx;
  renderStepStrip();
}

function failCurrentStep() {
  if (state.currentStepIdx >= 0) {
    state.stepStatuses[state.currentStepIdx] = "failed";
    renderStepStrip();
  }
}

function finalizeSteps(exitCode) {
  if (exitCode === 0) {
    for (let i = 0; i < STEPS.length; i++) {
      if (state.stepStatuses[i] !== "failed") state.stepStatuses[i] = "done";
    }
  } else {
    failCurrentStep();
  }
  renderStepStrip();
}

const STEP_RE = /^\[step (\d+)\]/;

function processLogLine(text) {
  const m = text.match(STEP_RE);
  if (m) { advanceToStep(parseInt(m[1], 10)); return; }
  if (/\[abort\]/.test(text)) failCurrentStep();
}

// ---------- source picker ----------

function hostnameOf(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch (e) { return ""; }
}

function renderPicker() {
  const list = $("picker-list");
  list.innerHTML = "";
  state.pickerResults.forEach((r) => {
    const row = document.createElement("label");
    row.className = "picker-row checked";
    row.dataset.url = r.url;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => {
      row.classList.toggle("checked", cb.checked);
      updatePickerMeta();
    });
    const body = document.createElement("div");
    const host = hostnameOf(r.url);
    const engine = r.engine ? `<span class="pr-engine">${r.engine}</span>` : "";
    body.innerHTML = `
      <div class="pr-title">${escapeHtml(r.title || host || r.url)}${engine}</div>
      <div class="pr-url">${escapeHtml(r.url)}</div>
      <div class="pr-snippet">${escapeHtml(r.snippet || "")}</div>
    `;
    row.appendChild(cb);
    row.appendChild(body);
    list.appendChild(row);
  });
  updatePickerMeta();
}

function selectedPickerUrls() {
  return Array.from(document.querySelectorAll("#picker-list .picker-row"))
    .filter((row) => row.querySelector("input[type=checkbox]").checked)
    .map((row) => row.dataset.url);
}

function updatePickerMeta() {
  const total = state.pickerResults.length;
  const selected = selectedPickerUrls().length;
  $("picker-meta").textContent = `${selected}/${total} selected`;
  $("picker-go").disabled = selected === 0 || !state.picking;
  $("picker-go").textContent = `Continue (${selected})`;
}

function showPicker(results) {
  state.pickerResults = results || [];
  state.picking = true;
  $("picker").hidden = false;
  $("picker").classList.remove("locked");
  renderPicker();
  setStatus(`paused — pick sources (${results.length} found)`);
}

function hidePicker() {
  state.picking = false;
  state.pickerResults = [];
  $("picker").hidden = true;
  $("picker").classList.remove("locked");
  $("picker-list").innerHTML = "";
}

function setAllChecked(checked) {
  document.querySelectorAll("#picker-list input[type=checkbox]").forEach((cb) => {
    cb.checked = checked;
    cb.closest(".picker-row").classList.toggle("checked", checked);
  });
  updatePickerMeta();
}

async function submitPicker() {
  if (!state.activeJobId || !state.picking) return;
  const urls = selectedPickerUrls();
  if (!urls.length) return;
  $("picker").classList.add("locked");
  setStatus(`resuming with ${urls.length} source(s)…`);
  try {
    const r = await fetch(`/api/research/${state.activeJobId}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      setStatus(`resume failed: ${err.detail || r.statusText}`);
      $("picker").classList.remove("locked");
    }
  } catch (e) {
    setStatus(`resume failed: ${e.message}`);
    $("picker").classList.remove("locked");
  }
}

// ---------- cascade picker ----------

function renderCascadeChips(containerId, names) {
  const el = $(containerId);
  el.innerHTML = "";
  for (const name of names) {
    const chip = document.createElement("label");
    chip.className = "cascade-chip";
    chip.dataset.name = name;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.addEventListener("change", () => {
      chip.classList.toggle("checked", cb.checked);
      updateCascadeMeta();
    });
    const span = document.createElement("span");
    span.textContent = name;
    chip.appendChild(cb);
    chip.appendChild(span);
    el.appendChild(chip);
  }
}

function selectedCascade(containerId) {
  return Array.from(document.querySelectorAll(`#${containerId} .cascade-chip`))
    .filter((c) => c.querySelector("input[type=checkbox]").checked)
    .map((c) => c.dataset.name);
}

function updateCascadeMeta() {
  const c = selectedCascade("cascade-children-list").length;
  const s = selectedCascade("cascade-siblings-list").length;
  const total = c + s;
  $("cascade-meta").textContent = `${total} selected`;
  $("cascade-go").disabled = total === 0;
  $("cascade-go").textContent = `Cascade selected (${total})`;
}

async function showCascadePicker(jobId, ctx) {
  const children = ctx.children || [];
  // Fetch siblings from the vault.
  let siblings = [];
  try {
    const r = await fetch(
      `/api/vault/siblings?topic=${encodeURIComponent(ctx.topic || "")}`
    );
    if (r.ok) {
      const data = await r.json();
      const childSet = new Set(children.map((c) => c.toLowerCase()));
      siblings = data
        .map((s) => s.filename_stem || s.title)
        .filter((s) => s && !childSet.has(s.toLowerCase()));
    }
  } catch (e) { /* siblings optional */ }

  state.cascadeJobId = jobId;
  state.cascadeContext = ctx;
  $("cascade-parent-label").textContent = ctx.topic_display || ctx.topic || "this topic";
  $("cascade-children-count").textContent = `(${children.length})`;
  $("cascade-siblings-count").textContent = `(${siblings.length})`;
  renderCascadeChips("cascade-children-list", children);
  renderCascadeChips("cascade-siblings-list", siblings);
  $("cascade-picker").hidden = false;
  updateCascadeMeta();
}

function hideCascadePicker() {
  state.cascadeJobId = null;
  state.cascadeContext = null;
  $("cascade-picker").hidden = true;
  $("cascade-children-list").innerHTML = "";
  $("cascade-siblings-list").innerHTML = "";
}

async function submitCascade() {
  if (!state.cascadeJobId) return;
  const children = selectedCascade("cascade-children-list");
  const siblings = selectedCascade("cascade-siblings-list");
  if (!children.length && !siblings.length) return;
  setStatus(`enqueuing ${children.length + siblings.length} cascade job(s)…`);
  try {
    const r = await fetch(`/api/research/${state.cascadeJobId}/cascade`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ children, siblings }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      setStatus(`cascade failed: ${err.detail || r.statusText}`);
      return;
    }
    const data = await r.json();
    setStatus(`enqueued ${data.queue_ids.length} cascade job(s)`);
    hideCascadePicker();
  } catch (e) {
    setStatus(`cascade failed: ${e.message}`);
  }
}

// ---------- request body builders ----------

function readSharedOptions() {
  const opts = {
    pause_pick_sources: $("opt-pause-pick").checked,
    dry_run: $("opt-dry-run").checked,
    no_parent: $("opt-no-parent").checked,
    cascade_picker: $("opt-cascade-picker").checked,
    cascade_depth: parseInt($("opt-cascade-depth").value, 10) || 0,
  };
  const maxStubs = $("opt-max-stubs").value.trim();
  if (maxStubs) opts.max_stubs = parseInt(maxStubs, 10);
  const topSources = $("opt-top-sources").value.trim();
  if (topSources) opts.top_sources = parseInt(topSources, 10);
  return opts;
}

function buildResearchBody() {
  return {
    query: $("query").value.trim(),
    provider: state.selectedProvider,
    ...readSharedOptions(),
  };
}

function buildMassBody() {
  const queries = massQueries().map((q) => ({ query: q }));
  const shared = {
    provider: state.selectedProvider,
    ...readSharedOptions(),
    pause_pick_sources: false, // mass mode auto-picks; toggle via single-mode if needed
  };
  return { queries, shared };
}

// ---------- start research (single + mass) ----------

async function startResearch() {
  if (state.activeJobId) return;
  const body = buildResearchBody();
  if (!body.query) return;

  prepareForNewJob(body.query);
  setStatus("starting job…");

  let resp;
  try {
    resp = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) { setStatus(`start failed: ${e.message}`); return; }
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    setStatus(`start failed: ${err.detail || resp.statusText}`);
    return;
  }
  const data = await resp.json();
  if (data.job_id) {
    setActiveJob(data.job_id, body.query);
  } else {
    // queued, awaiting concurrency slot — queue stream will tell us when started
    setStatus(`queued (queue id ${data.queue_id.slice(0, 8)})`);
  }
}

async function startMassResearch() {
  const body = buildMassBody();
  if (!body.queries.length) return;
  setStatus(`enqueuing ${body.queries.length} job(s)…`);
  try {
    const r = await fetch("/api/research/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      setStatus(`queue failed: ${err.detail || r.statusText}`);
      return;
    }
    const data = await r.json();
    const sk = data.skipped.length ? ` (${data.skipped.length} skipped)` : "";
    setStatus(`enqueued ${data.queue_ids.length} job(s)${sk}`);
    $("query-mass").value = "";
    updateMassButton();
  } catch (e) { setStatus(`queue failed: ${e.message}`); }
}

function prepareForNewJob(query) {
  clearLog();
  resetSteps();
  hidePicker();
  hideCascadePicker();
  $("log-title").textContent = `running: ${query}`;
}

function setActiveJob(jobId, label) {
  state.activeJobId = jobId;
  $("log-meta").textContent = `job ${jobId.slice(0, 8)}`;
  setStatus(`job ${jobId.slice(0, 8)} running${label ? `: ${label}` : ""}`);
  updateRunButton();
  openJobStream(jobId);
}

function clearActiveJob() {
  if (state.jobStream) {
    state.jobStream.close();
    state.jobStream = null;
  }
  state.activeJobId = null;
  updateRunButton();
}

async function maybeAttachToRunningJob() {
  if (state.activeJobId) return;
  if (!$("cascade-picker").hidden) return;  // user is picking; don't switch
  await refreshQueue();
  const next = (state.queue.running || [])[0];
  if (next && next.job_id) {
    prepareForNewJob(next.query || "(queued)");
    setActiveJob(next.job_id, next.query);
  }
}

// ---------- per-job SSE ----------

function openJobStream(jobId) {
  if (state.jobStream) {
    state.jobStream.close();
    state.jobStream = null;
  }
  const es = new EventSource(`/api/research/${jobId}/stream`);
  state.jobStream = es;

  es.addEventListener("line", (ev) => {
    let text = ev.data;
    try { text = JSON.parse(ev.data).text; } catch (e) {}
    appendLogLine(text);
    processLogLine(text);
  });

  es.addEventListener("pause", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      showPicker(data.results || []);
    } catch (e) { setStatus(`pause parse error: ${e.message}`); }
  });

  es.addEventListener("resumed", () => {
    hidePicker();
    setStatus("resumed");
  });

  es.addEventListener("cascade", (ev) => {
    try {
      const ctx = JSON.parse(ev.data);
      // Only show the picker if this option is on for the running job — we
      // can't tell server-side per-job, so we honor the global toggle here.
      if ($("opt-cascade-picker").checked) {
        showCascadePicker(jobId, ctx);
      }
    } catch (e) {}
  });

  es.addEventListener("done", async (ev) => {
    let exitCode = null;
    try { exitCode = JSON.parse(ev.data).exit_code; } catch (e) {}
    appendLogLine(
      exitCode === 0
        ? `\n[runner] job finished (exit 0)`
        : `\n[runner] job finished (exit ${exitCode})`
    );
    finalizeSteps(exitCode);
    setStatus(exitCode === 0 ? "job finished" : `job exited ${exitCode}`);
    hidePicker();
    es.close();
    state.jobStream = null;
    state.activeJobId = null;
    updateRunButton();
    await maybeAttachToRunningJob();
  });

  es.onerror = () => { setStatus("stream interrupted (retrying)"); };
}

// ---------- queue SSE + rendering ----------

function openQueueStream() {
  if (state.queueStream) state.queueStream.close();
  const es = new EventSource("/api/queue/stream");
  state.queueStream = es;

  es.addEventListener("snapshot", (ev) => {
    try {
      state.queue = JSON.parse(ev.data);
      renderQueuePanel();
    } catch (e) {}
  });

  es.addEventListener("job-enqueued", () => { refreshQueue(); });
  es.addEventListener("job-cancelled-pending", () => { refreshQueue(); });
  es.addEventListener("queue-cleared", () => { refreshQueue(); });
  es.addEventListener("concurrency-changed", () => { refreshQueue(); });

  es.addEventListener("job-started", (ev) => {
    refreshQueue();
    try {
      const data = JSON.parse(ev.data);
      const jobId = data.item && data.item.job_id;
      const q = data.item && data.item.query;
      // Don't trample the cascade picker — let the user finish picking before
      // we switch streams.
      const cascadeOpen = !$("cascade-picker").hidden;
      if (jobId && !state.activeJobId && !cascadeOpen) {
        prepareForNewJob(q || "(queued)");
        setActiveJob(jobId, q);
      }
    } catch (e) {}
  });

  es.addEventListener("job-finished", () => { refreshQueue(); });

  es.onerror = () => {};
}

async function refreshQueue() {
  try {
    const r = await fetch("/api/queue");
    if (!r.ok) return;
    state.queue = await r.json();
    renderQueuePanel();
  } catch (e) {}
}

function renderQueuePanel() {
  const q = state.queue;
  const visible = (q.pending && q.pending.length) || (q.running && q.running.length);
  $("queue-panel").hidden = !visible;
  if (!visible) return;
  $("queue-meta").textContent =
    `${q.pending.length} pending · ${q.running.length} running` +
    ` · ${q.recent_done.length} done · parallel ${q.concurrency}`;

  const list = $("queue-list");
  list.innerHTML = "";

  for (const item of q.running) {
    list.appendChild(makeQueueRow(item, "running", "▶"));
  }
  for (const item of q.pending) {
    list.appendChild(makeQueueRow(item, "pending", "⋯"));
  }
}

function makeQueueRow(item, kind, icon) {
  const li = document.createElement("li");
  li.className = `queue-item ${kind}`;
  const tags = [];
  if (item.parent_label) tags.push(item.parent_label);
  if (item.kwargs && item.kwargs.parent) tags.push(`parent: ${item.kwargs.parent}`);
  if (item.cascade_depth_remaining > 0) tags.push(`cascade ${item.cascade_depth_remaining}`);
  const tagsHtml = tags.map((t) => `<span class="qi-tag">${escapeHtml(t)}</span>`).join("");
  li.innerHTML = `
    <span class="qi-icon">${icon}</span>
    <span class="qi-query">${escapeHtml(item.query)}</span>
    <span class="qi-tags">${tagsHtml}</span>
  `;
  if (kind === "pending") {
    const btn = document.createElement("button");
    btn.className = "qi-cancel";
    btn.textContent = "×";
    btn.title = "Remove from queue";
    btn.addEventListener("click", () => cancelPending(item.queue_id));
    li.appendChild(btn);
  }
  return li;
}

async function cancelPending(queueId) {
  try {
    await fetch(`/api/queue/${queueId}`, { method: "DELETE" });
    refreshQueue();
  } catch (e) {}
}

async function clearQueue() {
  try {
    await fetch("/api/queue/clear", { method: "POST" });
    refreshQueue();
  } catch (e) {}
}

async function setConcurrency(n) {
  try {
    await fetch("/api/queue/concurrency", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n }),
    });
  } catch (e) {}
}

// ---------- cancel ----------

async function cancelCurrent() {
  if (!state.activeJobId) return;
  setStatus("cancelling…");
  try {
    const r = await fetch(`/api/research/${state.activeJobId}/cancel`, { method: "POST" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      setStatus(`cancel failed: ${err.detail || r.statusText}`);
    } else {
      setStatus("cancellation requested");
    }
  } catch (e) { setStatus(`cancel failed: ${e.message}`); }
}

// ---------- bootstrap ----------

document.addEventListener("DOMContentLoaded", async () => {
  renderStepStrip();
  await fetchState();
  await fetchProviders();
  openQueueStream();

  // Mode toggle
  $("mode-single").addEventListener("click", () => setMode("single"));
  $("mode-mass").addEventListener("click", () => setMode("mass"));

  // Provider strip
  $("refresh-providers").addEventListener("click", fetchProviders);

  // Single research
  $("query").addEventListener("input", updateRunButton);
  $("query").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !$("research-btn").disabled) startResearch();
  });
  $("research-btn").addEventListener("click", startResearch);
  $("cancel-btn").addEventListener("click", cancelCurrent);

  // Mass research
  $("query-mass").addEventListener("input", updateMassButton);
  $("queue-all-btn").addEventListener("click", startMassResearch);

  // Log
  $("log-clear").addEventListener("click", clearLog);

  // Source picker
  $("picker-go").addEventListener("click", submitPicker);
  $("picker-all").addEventListener("click", () => setAllChecked(true));
  $("picker-none").addEventListener("click", () => setAllChecked(false));

  // Cascade picker
  $("cascade-go").addEventListener("click", submitCascade);
  $("cascade-skip").addEventListener("click", hideCascadePicker);

  // Queue
  $("queue-clear").addEventListener("click", clearQueue);
  $("opt-concurrency").addEventListener("change", (e) => {
    const n = Math.max(1, Math.min(4, parseInt(e.target.value, 10) || 1));
    e.target.value = String(n);
    setConcurrency(n);
  });
});
