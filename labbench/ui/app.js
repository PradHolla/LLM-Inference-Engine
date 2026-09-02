// lab bench front end. No build step, no framework. State lives in memory only.
// ?mock=1 feeds every panel from fixtures in mockFixtures() -- see bottom of file.
"use strict";

(function () {

const MOCK = new URLSearchParams(location.search).get("mock") === "1";

const qs = (id) => document.getElementById(id);

// ---- formatting helpers -- every "may be null" field routes through these ----

function isNum(x) {
  return typeof x === "number" && !Number.isNaN(x);
}

function fmtNum(x, digits) {
  if (!isNum(x)) return "unavailable";
  return x.toFixed(digits == null ? 0 : digits);
}

function fmtPctFrac(x, digits) {
  if (!isNum(x)) return "unavailable";
  return (x * 100).toFixed(digits == null ? 1 : digits) + "%";
}

function fmtOr(x, suffix) {
  if (x === null || x === undefined || x === "") return "unavailable";
  return String(x) + (suffix || "");
}

function setText(el, text, na) {
  el.textContent = text;
  el.classList.toggle("na", !!na || text === "unavailable");
}

function setRow(id, value, digits, suffix) {
  const el = qs(id);
  if (!isNum(value)) { setText(el, "unavailable", true); return; }
  setText(el, fmtNum(value, digits) + (suffix || ""));
}

function withError(base, error) {
  return error ? base + " -- " + error : base;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ---- tiny markdown renderer. Input is raw untrusted text; output is safe HTML. ----
// Escapes first, then layers block and inline rules on the escaped text, so no
// entity produced by escaping is ever re-interpreted as markup.

function renderMarkdown(raw) {
  const escaped = escapeHtml(raw);
  const parts = escaped.split("```");
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      html += renderCodeBlock(parts[i]);
    } else {
      html += renderTextBlock(parts[i]);
    }
  }
  return html;
}

function renderCodeBlock(segment) {
  const m = segment.match(/^([\w+-]*)\n([\s\S]*)$/);
  const body = m ? m[2] : segment;
  return '<pre class="md-code"><code>' + body + "</code></pre>";
}

function renderTextBlock(segment) {
  const lines = segment.split("\n");
  let html = "";
  let list = null; // {tag, items: []}
  let para = [];

  function flushPara() {
    if (para.length) { html += "<p>" + para.join("<br>") + "</p>"; para = []; }
  }
  function flushList() {
    if (list) { html += "<" + list.tag + ">" + list.items.join("") + "</" + list.tag + ">"; list = null; }
  }

  for (const line of lines) {
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const ordered = line.match(/^\d+\.\s+(.*)$/);
    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (heading) {
      flushPara(); flushList();
      const n = heading[1].length;
      html += "<h" + n + ">" + inline(heading[2]) + "</h" + n + ">";
    } else if (bullet) {
      flushPara();
      if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; }
      list.items.push("<li>" + inline(bullet[1]) + "</li>");
    } else if (ordered) {
      flushPara();
      if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; }
      list.items.push("<li>" + inline(ordered[1]) + "</li>");
    } else if (line.trim() === "") {
      flushPara(); flushList();
    } else {
      flushList();
      para.push(inline(line));
    }
  }
  flushPara(); flushList();
  return html;
}

function inline(text) {
  let out = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\[([^\]]*)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, "$1<em>$2</em>");
  out = out.replace(/(^|[^_])_([^_]+)_(?!_)/g, "$1<em>$2</em>");
  return out;
}

// ---- app state ----

const App = {
  messages: [],          // {role, content, reasoningContent, streaming, turnIndex}
  turnTtfts: [],          // ttft_ms per assistant turn, null when unknown
  lastTrace: null,
  lastState: null,
  pinnedToBottom: true,
  streaming: false,
  abortCtrl: null,
  backendClicked: false,
};

// ---- network layer. All GET/POST for the labbench control API go through here. ----

async function apiGet(path) {
  if (MOCK) return mockGet(path);
  const res = await fetch(path);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

async function apiPost(path, body) {
  if (MOCK) return mockPost(path, body);
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

// ---- connection + state poll (500ms) ----

function setConnected(ok, errText) {
  const ind = qs("connIndicator");
  const banner = qs("connBanner");
  if (ok) {
    ind.textContent = "connected";
    ind.className = "conn-indicator conn-ok";
    banner.hidden = true;
  } else {
    ind.textContent = "disconnected";
    ind.className = "conn-indicator conn-error";
    banner.hidden = false;
    banner.textContent = "disconnected from labbench server -- retrying" + (errText ? " (" + errText + ")" : "");
  }
}

async function pollState() {
  try {
    const state = await apiGet("/labbench/state");
    App.lastState = state;
    setConnected(true);
    renderBackendControls(state.backend);
    renderConfigSummary(state.config);
    renderConfigPanel(state.config);
    renderEnginePanel(state.engine);
    renderGpuPanel(state.gpu);
    renderConversationPanel(state.config);
    if (activeTab === "gpu") renderGpuTabJson(state.gpu);
    if (activeTab === "server") renderServerInfo(state.backend, state.config);
    if (App.streaming) await pollLiveTrace();
  } catch (e) {
    setConnected(false, e.message);
  } finally {
    setTimeout(pollState, 500);
  }
}

// ---- top bar: config summary ----

function renderConfigSummary(config) {
  const el = qs("configSummary");
  if (!config) { setText(el, "unavailable", true); return; }
  const bits = [
    fmtOr(config.model), fmtOr(config.quantization), "ctx=" + fmtOr(config.max_model_len),
    "kv=" + fmtOr(config.kv_gib, "GiB"), "spec=" + fmtOr(config.spec),
  ];
  setText(el, bits.join("  |  "));
}

// ---- top bar: backend controls. Selection reflects the poll, never the click. ----

const BACKEND_VALUES = ["vllm", "baseline", "engine"];

function renderBackendControls(backend) {
  const group = qs("backendGroup");
  const quantSelect = qs("quantSelect");
  const statusEl = qs("switchStatus");
  const active = backend ? backend.active : null;
  const status = backend ? backend.status : null;
  const switching = status === "switching";

  for (const v of BACKEND_VALUES) {
    const radio = qs("backend-" + v);
    radio.checked = status !== "failed" && active === v;
    radio.disabled = switching;
  }
  group.disabled = switching;
  quantSelect.disabled = switching;
  // Same rule as the radio: show what is running, not what was clicked.
  if (!switching && App.lastState && App.lastState.config &&
      App.lastState.config.quantization && document.activeElement !== quantSelect) {
    quantSelect.value = App.lastState.config.quantization;
  }

  if (switching) {
    statusEl.className = "switch-status";
    statusEl.textContent = "switching: " + fmtOr(backend.stage) + " (" + fmtNum(backend.elapsed_s, 1) + "s)";
  } else if (status === "failed") {
    statusEl.className = "switch-status error";
    statusEl.textContent = "failed: " + fmtOr(backend.error);
  } else if (status === "stopped" || active === null) {
    statusEl.className = "switch-status";
    statusEl.textContent = active === null ? "no backend active" : "";
  } else {
    statusEl.className = "switch-status";
    statusEl.textContent = "";
  }

  const sendDisabled = switching || active === null;
  qs("sendBtn").disabled = sendDisabled || App.streaming;
  qs("chatInput").disabled = sendDisabled;
}

function initBackendControls() {
  for (const v of BACKEND_VALUES) {
    const radio = qs("backend-" + v);
    radio.addEventListener("click", (e) => {
      e.preventDefault();
      if (radio.disabled) return;
      const quantization = v === "vllm" ? qs("quantSelect").value : null;
      apiPost("/labbench/backend", { backend: v, quantization })
        .catch((err) => console.error("backend switch failed", err));
    });
  }
}

// ---- panel 1: this request ----

function renderRequestPanelPending() {
  setText(qs("req-ttft"), "pending", true);
  setText(qs("req-ttft-pred"), "unavailable", true);
  setText(qs("req-ttft-gap"), "unavailable", true);
}

function renderRequestPanelLive(clientTtftMs) {
  setRow("req-ttft-client", clientTtftMs, 0, " ms");
}

function renderRequestPanelFromTrace(trace, prediction) {
  if (!trace) { renderRequestPanelPending(); return; }
  setRow("req-ttft", trace.ttft_ms, 0);
  if (trace.ttft_ms === null || trace.ttft_ms === undefined) setText(qs("req-ttft"), "unavailable", true);

  const predVal = prediction && prediction.values ? prediction.values.ttft_ms : null;
  if (isNum(predVal)) {
    setText(qs("req-ttft-pred"), fmtNum(predVal, 0));
    if (isNum(trace.ttft_ms) && predVal !== 0) {
      const gap = ((trace.ttft_ms - predVal) / predVal) * 100;
      setText(qs("req-ttft-gap"), (gap >= 0 ? "+" : "") + gap.toFixed(1) + "%");
    } else {
      setText(qs("req-ttft-gap"), "unavailable", true);
    }
  } else {
    setText(qs("req-ttft-pred"), withError("unavailable", prediction && prediction.error), true);
    setText(qs("req-ttft-gap"), "unavailable", true);
  }

  const interevent = (isNum(trace.inter_event_p50_ms) || isNum(trace.inter_event_p95_ms))
    ? fmtNum(trace.inter_event_p50_ms, 1) + " / " + fmtNum(trace.inter_event_p95_ms, 1) + " ms"
    : "unavailable";
  setText(qs("req-interevent"), interevent, interevent === "unavailable");
  setRow("req-itl", trace.itl_ms_derived, 1, " ms");
  setRow("req-tpe", trace.tokens_per_event, 2);
  setRow("req-prompt-tokens", trace.prompt_tokens, 0);
  setRow("req-cached-tokens", trace.cached_tokens, 0);
  setRow("req-completion-tokens", trace.completion_tokens, 0);
}

// ---- panel 2: engine now ----

function renderEnginePanel(engine) {
  const errEl = qs("eng-error");
  if (!engine || engine.error) {
    setText(errEl, engine && engine.error ? engine.error : "");
    setText(qs("eng-running"), "unavailable", true);
    setText(qs("eng-waiting"), "unavailable", true);
    setText(qs("eng-kv-pct"), "unavailable", true);
    qs("eng-kv-bar").style.width = "0%";
    qs("eng-kv-bar").className = "bar-fill";
    setText(qs("eng-prefix-hit"), "unavailable", true);
    setText(qs("eng-bound"), "unavailable", true);
    return;
  }
  setText(errEl, "");
  const v = engine.values || {};
  setRow("eng-running", v.running, 0);
  setRow("eng-waiting", v.waiting, 0);
  if (isNum(v.kv_usage)) {
    setText(qs("eng-kv-pct"), fmtPctFrac(v.kv_usage, 1));
    qs("eng-kv-bar").style.width = Math.min(100, v.kv_usage * 100) + "%";
    qs("eng-kv-bar").className = "bar-fill state-normal";
  } else {
    setText(qs("eng-kv-pct"), "unavailable", true);
    qs("eng-kv-bar").style.width = "0%";
  }
  setText(qs("eng-prefix-hit"), isNum(v.prefix_hit_rate) ? fmtPctFrac(v.prefix_hit_rate, 1) : "unavailable", !isNum(v.prefix_hit_rate));
  setText(qs("eng-bound"), engine.bound ? JSON.stringify(engine.bound, null, 2) : "unavailable", !engine.bound);
}

// ---- panel 3: gpu now ----

function renderGpuPanel(gpu) {
  const errEl = qs("gpu-error");
  if (!gpu || gpu.error || !gpu.devices || !gpu.devices.length) {
    setText(errEl, gpu && gpu.error ? gpu.error : (gpu ? "no devices reported" : ""));
    setText(qs("gpu-mem-text"), "unavailable", true);
    setText(qs("gpu-power-text"), "unavailable", true);
    setText(qs("gpu-util"), "unavailable", true);
    setText(qs("gpu-temp"), "unavailable", true);
    qs("gpu-mem-bar").style.width = "0%";
    qs("gpu-power-bar").style.width = "0%";
    qs("gpu-proc-body").innerHTML = "";
    return;
  }
  setText(errEl, "");
  const d = gpu.devices[0];
  if (isNum(d.memory_used) && isNum(d.memory_total) && d.memory_total > 0) {
    const pct = (d.memory_used / d.memory_total) * 100;
    setText(qs("gpu-mem-text"), fmtNum(d.memory_used, 0) + " / " + fmtNum(d.memory_total, 0) + " MiB (" + pct.toFixed(1) + "%)");
    qs("gpu-mem-bar").style.width = Math.min(100, pct) + "%";
  } else {
    setText(qs("gpu-mem-text"), "unavailable", true);
    qs("gpu-mem-bar").style.width = "0%";
  }
  if (isNum(d.power_draw) && isNum(d.power_limit) && d.power_limit > 0) {
    const pct = (d.power_draw / d.power_limit) * 100;
    setText(qs("gpu-power-text"), fmtNum(d.power_draw, 0) + " / " + fmtNum(d.power_limit, 0) + " W");
    qs("gpu-power-bar").style.width = Math.min(100, pct) + "%";
  } else {
    setText(qs("gpu-power-text"), "unavailable", true);
    qs("gpu-power-bar").style.width = "0%";
  }
  setRow("gpu-util", d.utilization_gpu, 0, "%");
  setRow("gpu-temp", d.temperature_gpu, 0, " C");

  const body = qs("gpu-proc-body");
  body.innerHTML = "";
  const procs = gpu.processes || [];
  if (!procs.length) {
    body.innerHTML = '<tr><td colspan="2" class="na">unavailable</td></tr>';
  } else {
    for (const p of procs) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + fmtOr(p.pid) + "</td><td>" + fmtNum(p.used_mib, 0) + "</td>";
      body.appendChild(tr);
    }
  }
}

function renderGpuTabJson(gpu) {
  setText(qs("gpuJson"), gpu ? JSON.stringify(gpu, null, 2) : "unavailable", !gpu);
}

// ---- panel 4: this conversation ----

function renderConversationPanel(config) {
  const turns = App.messages.filter((m) => m.role === "assistant").length;
  setText(qs("conv-turns"), String(turns));

  const promptTokens = App.lastTrace ? App.lastTrace.prompt_tokens : null;
  const maxLen = config ? config.max_model_len : null;
  if (isNum(promptTokens) && isNum(maxLen) && maxLen > 0) {
    const pct = (promptTokens / maxLen) * 100;
    setText(qs("conv-context-text"), promptTokens + " / " + maxLen + " (" + pct.toFixed(1) + "%)");
    const bar = qs("conv-context-bar");
    bar.style.width = Math.min(100, pct) + "%";
    bar.className = "bar-fill " + (pct > 95 ? "state-error" : pct > 80 ? "state-warning" : "state-normal");
  } else {
    setText(qs("conv-context-text"), "unavailable", true);
    qs("conv-context-bar").style.width = "0%";
    qs("conv-context-bar").className = "bar-fill";
  }

  renderTtftChart();
}

function renderTtftChart() {
  const el = qs("conv-ttft-chart");
  el.innerHTML = "";
  const vals = App.turnTtfts;
  if (!vals.length) return;
  const max = Math.max(1, ...vals.filter(isNum));
  for (const v of vals) {
    const bar = document.createElement("div");
    if (isNum(v)) {
      bar.className = "ttft-bar";
      bar.style.height = Math.max(2, (v / max) * 100) + "%";
      bar.title = v.toFixed(0) + " ms";
    } else {
      bar.className = "ttft-bar na";
      bar.style.height = "100%";
      bar.title = "unavailable";
    }
    el.appendChild(bar);
  }
}

// ---- panel 5: config ----

function renderConfigPanel(config) {
  if (!config) {
    for (const id of ["cfg-model", "cfg-quant", "cfg-max-model-len", "cfg-kv-tokens", "cfg-kv-gib", "cfg-spec"]) {
      setText(qs(id), "unavailable", true);
    }
    return;
  }
  setText(qs("cfg-model"), fmtOr(config.model), config.model == null);
  setText(qs("cfg-quant"), fmtOr(config.quantization), config.quantization == null);
  setRow("cfg-max-model-len", config.max_model_len, 0);
  setRow("cfg-kv-tokens", config.kv_tokens, 0);
  setRow("cfg-kv-gib", config.kv_gib, 2, " GiB");
  setText(qs("cfg-spec"), fmtOr(config.spec), config.spec == null);
}

// ---- chat: send + stream ----

function scrollAnchorInit() {
  const container = qs("chatMessages");
  container.addEventListener("scroll", () => {
    const dist = container.scrollHeight - container.scrollTop - container.clientHeight;
    App.pinnedToBottom = dist < 40;
    qs("jumpLatest").hidden = App.pinnedToBottom;
  });
  qs("jumpLatest").addEventListener("click", () => {
    container.scrollTop = container.scrollHeight;
    App.pinnedToBottom = true;
    qs("jumpLatest").hidden = true;
  });
}

function maybeScrollToBottom() {
  if (!App.pinnedToBottom) return;
  const container = qs("chatMessages");
  container.scrollTop = container.scrollHeight;
}

function createMessageEl(msg) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-" + msg.role;
  const roleEl = document.createElement("div");
  roleEl.className = "msg-role";
  roleEl.textContent = msg.role;
  wrap.appendChild(roleEl);

  if (msg.role === "assistant") {
    const thinking = document.createElement("details");
    thinking.className = "thinking-block";
    const summary = document.createElement("summary");
    summary.textContent = "thinking (0 chars)";
    const content = document.createElement("div");
    content.className = "thinking-content";
    thinking.appendChild(summary);
    thinking.appendChild(content);
    thinking.hidden = true;
    wrap.appendChild(thinking);
    msg._thinkingEl = thinking;
    msg._thinkingSummary = summary;
    msg._thinkingContent = content;
  }

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  wrap.appendChild(bubble);
  msg._bubbleEl = bubble;
  msg._wrapEl = wrap;
  return wrap;
}

function renderMessageContent(msg) {
  if (msg.role === "assistant") {
    msg._bubbleEl.innerHTML = renderMarkdown(msg.content || "");
    if (msg.reasoningContent) {
      msg._thinkingEl.hidden = false;
      msg._thinkingSummary.textContent = "thinking (" + msg.reasoningContent.length + " chars)";
      msg._thinkingContent.textContent = msg.reasoningContent;
    }
  } else {
    msg._bubbleEl.textContent = msg.content || "";
  }
}

function appendMessage(role, content) {
  const msg = { role, content: content || "", reasoningContent: "", streaming: false };
  App.messages.push(msg);
  const el = createMessageEl(msg);
  qs("chatMessages").appendChild(el);
  renderMessageContent(msg);
  maybeScrollToBottom();
  return msg;
}

function buildApiMessages() {
  return App.messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role, content: m.content || "" }));
}

async function sendMessage() {
  const input = qs("chatInput");
  const text = input.value.trim();
  if (!text || App.streaming) return;
  const backend = App.lastState && App.lastState.backend;
  if (!backend || backend.status === "switching" || backend.active === null) return;

  appendMessage("user", text);
  input.value = "";
  const userMessages = buildApiMessages();
  const assistantMsg = appendMessage("assistant", "");
  assistantMsg.streaming = true;
  App.streaming = true;
  App.lastTrace = null;
  renderRequestPanelPending();
  qs("sendBtn").disabled = true;
  qs("stopBtn").hidden = false;

  const thinkingEnabled = qs("thinkingToggle").checked;
  const clientStart = performance.now();
  let firstContentAt = null;

  function applyDelta(delta) {
    if (delta.reasoning_content) {
      assistantMsg.reasoningContent += delta.reasoning_content;
      renderMessageContent(assistantMsg);
    }
    if (delta.content) {
      if (firstContentAt === null) {
        firstContentAt = performance.now();
        renderRequestPanelLive(firstContentAt - clientStart);
      }
      assistantMsg.content += delta.content;
      renderMessageContent(assistantMsg);
    }
    maybeScrollToBottom();
  }

  function finalize() {
    assistantMsg.streaming = false;
    App.streaming = false;
    App.abortCtrl = null;
    qs("stopBtn").hidden = true;
    if (App.lastState) qs("sendBtn").disabled = false;
    finalizeTrace();
  }

  if (MOCK) {
    await mockStreamChat(applyDelta);
    finalize();
    return;
  }

  const ctrl = new AbortController();
  App.abortCtrl = ctrl;
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: ctrl.signal,
      body: JSON.stringify({
        model: (App.lastState && App.lastState.config && App.lastState.config.served_model) || "labbench",
        messages: userMessages,
        stream: true,
        stream_options: { include_usage: true },
        max_tokens: 1024,
        temperature: 0.0,
        chat_template_kwargs: { enable_thinking: thinkingEnabled },
      }),
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        let ev;
        try { ev = JSON.parse(data); } catch { continue; }
        const choice = (ev.choices || [])[0];
        if (choice && choice.delta) applyDelta(choice.delta);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      assistantMsg.content += "\n\n[error: " + e.message + "]";
      renderMessageContent(assistantMsg);
    } else {
      assistantMsg.content += "\n\n[stopped]";
      renderMessageContent(assistantMsg);
    }
  } finally {
    finalize();
  }
}

async function pollLiveTrace() {
  // The server records TTFT on the first content event, so it is available mid-stream.
  // Prediction is deliberately not fetched here: prompt_tokens is only final at the end.
  try {
    const data = await apiGet("/labbench/traces?n=1");
    const list = data.traces || [];
    const live = list.length ? list[list.length - 1] : null;
    if (live && live.status === "streaming") renderRequestPanelFromTrace(live, null);
  } catch (e) {
    /* the state poll already reports connection loss */
  }
}

async function finalizeTrace() {
  try {
    const data = await apiGet("/labbench/traces?n=1");
    const trace = data.traces && data.traces.length ? data.traces[data.traces.length - 1] : null;
    App.lastTrace = trace;
    App.turnTtfts.push(trace && isNum(trace.ttft_ms) ? trace.ttft_ms : null);
    let prediction = null;
    if (trace && isNum(trace.prompt_tokens)) {
      try { prediction = await apiGet("/labbench/prediction?context=" + trace.prompt_tokens); }
      catch (e) { prediction = { error: e.message, values: {} }; }
    }
    renderRequestPanelFromTrace(trace, prediction);
    renderWireTab(trace);
    renderConversationPanel(App.lastState ? App.lastState.config : null);
  } catch (e) {
    App.turnTtfts.push(null);
    renderRequestPanelFromTrace(null, null);
  }
}

function stopStreaming() {
  if (App.abortCtrl) App.abortCtrl.abort();
}

// ---- bottom drawer ----

let activeTab = "prompt";

function initQuantSelect() {
  const sel = qs("quantSelect");
  sel.addEventListener("change", async () => {
    const st = App.lastState && App.lastState.backend;
    if (!st || st.active !== "vllm" || st.status === "switching") return;
    try { await apiPost("/labbench/backend", { backend: "vllm", quantization: sel.value }); }
    catch (e) { /* the poll reports the outcome */ }
  });
}

function initDrawer() {
  qs("drawerToggle").addEventListener("click", () => {
    qs("drawer").classList.toggle("open");
  });
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.addEventListener("click", () => {
      for (const b of document.querySelectorAll(".tab-btn")) b.classList.remove("active");
      for (const p of document.querySelectorAll(".tab-panel")) p.classList.remove("active");
      btn.classList.add("active");
      activeTab = btn.dataset.tab;
      qs("tab-" + activeTab).classList.add("active");
      onTabActivated(activeTab);
    });
  }
}

function onTabActivated(tab) {
  if (tab === "prompt") refreshPrompt();
  if (tab === "metrics") refreshMetrics();
  if (tab === "gpu" && App.lastState) renderGpuTabJson(App.lastState.gpu);
  if (tab === "server" && App.lastState) renderServerInfo(App.lastState.backend, App.lastState.config);
  if (tab === "wire") renderWireTab(App.lastTrace);
}

// -- prompt tab --

async function refreshPrompt() {
  try {
    const data = await apiPost("/labbench/render", {
      messages: buildApiMessages(),
      enable_thinking: qs("thinkingToggle").checked,
    });
    if (data.error) {
      setText(qs("promptOutput"), "unavailable -- " + data.error, true);
      setText(qs("promptNTokens"), "unavailable", true);
    } else {
      setText(qs("promptOutput"), fmtOr(data.prompt), data.prompt == null);
      setRow("promptNTokens", data.n_tokens, 0);
    }
  } catch (e) {
    setText(qs("promptOutput"), "unavailable -- " + e.message, true);
    setText(qs("promptNTokens"), "unavailable", true);
  }
}

// -- wire tab --

function renderWireTab(trace) {
  const body = qs("wireBody");
  body.innerHTML = "";
  if (!trace || !trace.events) {
    setText(qs("wireSummary"), "unavailable", true);
    return;
  }
  setText(qs("wireSummary"),
    "n_events=" + fmtOr(trace.n_events) + "  n_content_events=" + fmtOr(trace.n_content_events) +
    "  tokens_per_event=" + (isNum(trace.tokens_per_event) ? trace.tokens_per_event.toFixed(2) : "unavailable"));
  trace.events.forEach((ev, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + i + "</td><td>" + fmtOr(ev.t_ms) + "</td><td>" + fmtOr(ev.bytes) + "</td><td>" + fmtOr(ev.chars) + "</td>";
    body.appendChild(tr);
  });
}

// -- metrics tab --

let metricsRaw = "";

async function refreshMetrics() {
  try {
    const data = await apiGet("/labbench/metrics/raw");
    if (data.error) {
      metricsRaw = "";
      setText(qs("metricsOutput"), "unavailable -- " + data.error, true);
    } else {
      metricsRaw = data.text || "";
      applyMetricsFilter();
    }
  } catch (e) {
    metricsRaw = "";
    setText(qs("metricsOutput"), "unavailable -- " + e.message, true);
  }
}

function applyMetricsFilter() {
  const filter = qs("metricsFilter").value.toLowerCase();
  if (!filter) { setText(qs("metricsOutput"), metricsRaw || "unavailable", !metricsRaw); return; }
  const lines = metricsRaw.split("\n").filter((l) => l.toLowerCase().includes(filter));
  setText(qs("metricsOutput"), lines.join("\n"));
}

// -- server tab --

function renderServerInfo(backend, config) {
  const info = {
    backend: backend || "unavailable",
    launch_cmd: config ? fmtOr(config.launch_cmd) : "unavailable",
  };
  setText(qs("serverInfo"), JSON.stringify(info, null, 2));
}

let journalTimer = null;

async function pollJournal() {
  try {
    const data = await apiGet("/labbench/journal?lines=200");
    const pre = qs("serverJournal");
    if (data.error) {
      setText(pre, "unavailable -- " + data.error, true);
    } else {
      setText(pre, (data.lines || []).join("\n"));
      pre.scrollTop = pre.scrollHeight;
    }
  } catch (e) {
    setText(qs("serverJournal"), "unavailable -- " + e.message, true);
  } finally {
    const switching = App.lastState && App.lastState.backend && App.lastState.backend.status === "switching";
    journalTimer = setTimeout(pollJournal, switching ? 2000 : 10000);
  }
}

// -- bombard tab --

let bombardTimer = null;

async function runBombard() {
  const n = parseInt(qs("bombardN").value, 10) || 8;
  const promptTokens = parseInt(qs("bombardPromptTokens").value, 10) || 512;
  const maxTokens = parseInt(qs("bombardMaxTokens").value, 10) || 64;
  qs("bombardRun").disabled = true;
  setText(qs("bombardOutput"), "starting...");
  try {
    await apiPost("/labbench/bombard", { n, prompt_tokens: promptTokens, max_tokens: maxTokens });
    pollBombard();
  } catch (e) {
    setText(qs("bombardOutput"), "unavailable -- " + e.message, true);
    qs("bombardRun").disabled = false;
  }
}

async function pollBombard() {
  try {
    const data = await apiGet("/labbench/bombard");
    setText(qs("bombardOutput"), fmtOr(data.stdout), data.stdout == null);
    if (data.running) {
      bombardTimer = setTimeout(pollBombard, 1000);
    } else {
      qs("bombardRun").disabled = false;
    }
  } catch (e) {
    setText(qs("bombardOutput"), "unavailable -- " + e.message, true);
    qs("bombardRun").disabled = false;
  }
}

// ---- wiring ----

function init() {
  scrollAnchorInit();
  initBackendControls();
  initQuantSelect();
  initDrawer();

  qs("sendBtn").addEventListener("click", sendMessage);
  qs("stopBtn").addEventListener("click", stopStreaming);
  qs("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  qs("promptRefresh").addEventListener("click", refreshPrompt);
  qs("metricsRefresh").addEventListener("click", refreshMetrics);
  qs("metricsFilter").addEventListener("input", applyMetricsFilter);
  qs("bombardRun").addEventListener("click", runBombard);

  pollState();
  pollJournal();
  refreshPrompt();

  if (MOCK) initMockUi();
}

document.addEventListener("DOMContentLoaded", init);

// ==== MOCK MODE (?mock=1) ====================================================
// Every fixture the UI can consume lives in this one function. apiGet/apiPost
// route here instead of fetch() when MOCK is true, and sendMessage() calls
// mockStreamChat() instead of opening a real SSE connection.

function mockFixtures() {
  return {
    state: {
      backend: { active: "vllm", status: "ready", stage: null, elapsed_s: 0, error: null },
      config: {
        model: "Qwen/Qwen3-8B", quantization: "fp8", max_model_len: 16384,
        kv_tokens: 68592, kv_gib: 9.42, spec: "eagle3(k=3)",
        launch_cmd: "vllm serve Qwen/Qwen3-8B --quantization fp8 --speculative-config ...",
      },
      engine: {
        values: { running: 2, waiting: 0, kv_usage: 0.412, prefix_hit_rate: 0.83 },
        bound: {
          running: ["vllm:num_requests_running"], waiting: ["vllm:num_requests_waiting"],
          kv_usage: ["vllm:gpu_cache_usage_perc"], prefix_queries: ["vllm:gpu_prefix_cache_queries_total"],
          prefix_hits: ["vllm:gpu_prefix_cache_hits_total"],
        },
        error: null,
      },
      gpu: {
        devices: [{ memory_total: 23028, memory_used: 15628, utilization_gpu: 74,
                    power_draw: 231.4, power_limit: 300.0, temperature_gpu: 68 }],
        processes: [{ pid: 3412, used_mib: 15320 }, { pid: 3999, used_mib: 308 }],
        error: null,
      },
    },
    traces: {
      upstream: "http://localhost:8000",
      traces: [{
        request_id: "mock0001", status: "ok", http_status: 200, error: null,
        ttft_ms: 187.3, e2e_ms: 4210.5, n_events: 340, n_content_events: 112,
        prompt_tokens: 812, cached_tokens: 640, completion_tokens: 256,
        tokens_per_event: 2.29, inter_event_p50_ms: 28.4, inter_event_p95_ms: 61.2,
        itl_ms_derived: 15.8,
        events: [{ t_ms: 187.3, bytes: 64, chars: 4 }, { t_ms: 215.7, bytes: 71, chars: 9 },
                 { t_ms: 244.1, bytes: 58, chars: 3 }],
      }],
    },
    render: { prompt: "<|im_start|>user\nmock prompt text<|im_end|>\n<|im_start|>assistant\n", n_tokens: 812, error: null },
    metrics: { text: "vllm:num_requests_running 2.0\nvllm:num_requests_waiting 0.0\nvllm:gpu_cache_usage_perc 0.412\n", error: null },
    journal: { unit: "vllm.service", lines: ["INFO 09-01 10:22:14 starting vllm", "INFO 09-01 10:22:41 model loaded", "INFO 09-01 10:22:42 ready"], error: null },
    prediction: { values: { ttft_ms: 172.0 }, error: null },
    bombard: { job: null, running: false, stdout: "", returncode: null },
  };
}

function mockGet(path) {
  const f = mockFixtures();
  if (path.startsWith("/labbench/state")) return Promise.resolve(f.state);
  if (path.startsWith("/labbench/traces")) return Promise.resolve(f.traces);
  if (path.startsWith("/labbench/metrics/raw")) return Promise.resolve(f.metrics);
  if (path.startsWith("/labbench/journal")) return Promise.resolve(f.journal);
  if (path.startsWith("/labbench/prediction")) return Promise.resolve(f.prediction);
  if (path.startsWith("/labbench/bombard")) return Promise.resolve(f.bombard);
  return Promise.resolve({});
}

function mockPost(path, body) {
  const f = mockFixtures();
  if (path.startsWith("/labbench/render")) return Promise.resolve(f.render);
  if (path.startsWith("/labbench/backend")) return Promise.resolve({ accepted: true, error: null });
  if (path.startsWith("/labbench/bombard")) return Promise.resolve({ job: "mock-job" });
  return Promise.resolve({});
}

function mockStreamChat(applyDelta) {
  const reasoning = "Considering the mock request and how a reasoning model would think about it.";
  const content = "This is a **mocked** streamed reply.\n\n- point one\n- point two\n\n```py\nprint('hello')\n```";
  return new Promise((resolve) => {
    let ri = 0, ci = 0;
    const step = () => {
      if (ri < reasoning.length) {
        applyDelta({ reasoning_content: reasoning.slice(ri, ri + 4) });
        ri += 4;
        setTimeout(step, 20);
      } else if (ci < content.length) {
        applyDelta({ content: content.slice(ci, ci + 4) });
        ci += 4;
        setTimeout(step, 20);
      } else {
        resolve();
      }
    };
    step();
  });
}

function initMockUi() {
  appendMessage("user", "what is speculative decoding");
  const reply = appendMessage("assistant", "Speculative decoding drafts several tokens with a small model and verifies them in one batch with the large model.");
  reply.reasoningContent = "The user is asking for a short definition; keep it to one sentence.";
  renderMessageContent(reply);
  App.turnTtfts.push(187.3, 203.9, 165.0);
  App.lastTrace = mockFixtures().traces.traces[0];
}

})();
