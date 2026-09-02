// lab bench front end -- React port of the original vanilla-JS app. No build step:
// htm.bind(React.createElement) gives JSX-like templates with no transpile.
// ?mock=1 feeds every panel from mockFixtures() -- see the MOCK MODE section at the bottom.
"use strict";

(function () {

const {
  useState, useEffect, useRef, useMemo, useCallback,
  memo, forwardRef, useImperativeHandle, Fragment,
} = React;

const html = htm.bind(React.createElement);

const MOCK = new URLSearchParams(location.search).get("mock") === "1";

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

function rowText(value, digits, suffix) {
  return isNum(value) ? fmtNum(value, digits) + (suffix || "") : "unavailable";
}

function withError(base, error) {
  return error ? base + " -- " + error : base;
}

// ---- markdown -> React elements. Input is raw untrusted text. No HTML string is
// ever built; every value reaches the DOM as a React child, so it is escaped by
// React the same way any other text child would be. Never dangerouslySetInnerHTML.

function applyInline(nodes, regex, keyBase, build) {
  const out = [];
  let counter = 0;
  for (const node of nodes) {
    if (typeof node !== "string") { out.push(node); continue; }
    regex.lastIndex = 0;
    let lastIndex = 0;
    let m;
    while ((m = regex.exec(node))) {
      if (m.index > lastIndex) out.push(node.slice(lastIndex, m.index));
      const built = build(m, keyBase + "-" + counter++);
      if (built.prefix) out.push(built.prefix);
      out.push(built.element);
      lastIndex = m.index + m[0].length;
      if (m[0].length === 0) regex.lastIndex++;
    }
    if (lastIndex < node.length) out.push(node.slice(lastIndex));
  }
  return out;
}

function inlineElements(text, keyBase) {
  let nodes = [text];
  nodes = applyInline(nodes, /`([^`]+)`/g, keyBase + "-code", (m, k) =>
    ({ element: html`<code key=${k}>${m[1]}</code>` }));
  nodes = applyInline(nodes, /\[([^\]]*)\]\(([^)\s]+)\)/g, keyBase + "-link", (m, k) =>
    ({ element: html`<a key=${k} href=${m[2]} target="_blank" rel="noopener">${m[1]}</a>` }));
  nodes = applyInline(nodes, /\*\*([^*]+)\*\*/g, keyBase + "-b", (m, k) =>
    ({ element: html`<strong key=${k}>${m[1]}</strong>` }));
  nodes = applyInline(nodes, /(^|[^*])\*([^*]+)\*(?!\*)/g, keyBase + "-i1", (m, k) =>
    ({ prefix: m[1], element: html`<em key=${k}>${m[2]}</em>` }));
  nodes = applyInline(nodes, /(^|[^_])_([^_]+)_(?!_)/g, keyBase + "-i2", (m, k) =>
    ({ prefix: m[1], element: html`<em key=${k}>${m[2]}</em>` }));
  return nodes;
}

function renderCodeBlockEl(segment, key) {
  const m = segment.match(/^([\w+-]*)\n([\s\S]*)$/);
  const body = m ? m[2] : segment;
  return html`<pre class="md-code" key=${key}><code>${body}</code></pre>`;
}

function renderTextBlockEls(segment, keyBase) {
  const lines = segment.split("\n");
  const blocks = [];
  let list = null;
  let para = [];
  let blockKey = 0;

  function flushPara() {
    if (!para.length) return;
    const children = [];
    para.forEach((lineNodes, idx) => {
      if (idx > 0) children.push(html`<br key=${keyBase + "-br" + blockKey + "-" + idx} />`);
      children.push(...lineNodes);
    });
    blocks.push(html`<p key=${keyBase + "-p" + blockKey++}>${children}</p>`);
    para = [];
  }
  function flushList() {
    if (!list) return;
    const Tag = list.tag;
    blocks.push(html`<${Tag} key=${keyBase + "-l" + blockKey++}>${list.items}</${Tag}>`);
    list = null;
  }

  lines.forEach((line, li) => {
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const ordered = line.match(/^\d+\.\s+(.*)$/);
    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (heading) {
      flushPara(); flushList();
      const Htag = "h" + heading[1].length;
      blocks.push(html`<${Htag} key=${keyBase + "-h" + blockKey++}>${inlineElements(heading[2], keyBase + "-h" + li)}</${Htag}>`);
    } else if (bullet) {
      flushPara();
      if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; }
      list.items.push(html`<li key=${keyBase + "-li" + li}>${inlineElements(bullet[1], keyBase + "-li" + li)}</li>`);
    } else if (ordered) {
      flushPara();
      if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; }
      list.items.push(html`<li key=${keyBase + "-li" + li}>${inlineElements(ordered[1], keyBase + "-li" + li)}</li>`);
    } else if (line.trim() === "") {
      flushPara(); flushList();
    } else {
      flushList();
      para.push(inlineElements(line, keyBase + "-ln" + li));
    }
  });
  flushPara(); flushList();
  return blocks;
}

function renderMarkdownElements(raw) {
  const parts = (raw || "").split("```");
  const blocks = [];
  parts.forEach((part, i) => {
    if (i % 2 === 1) blocks.push(renderCodeBlockEl(part, "code" + i));
    else blocks.push(...renderTextBlockEls(part, "blk" + i));
  });
  return blocks;
}

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

function buildApiMessages(messages) {
  return messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role, content: m.content || "" }));
}

// ---- small reusable presentational pieces ----

function MetricRow({ label, text }) {
  const na = text === "unavailable";
  return html`
    <div class="metric-row">
      <span class="metric-row-label">${label}</span>
      <span class="metric-row-value${na ? " na" : ""}">${text}</span>
    </div>`;
}

function BarRow({ label, text, pct, fillClass }) {
  const na = text === "unavailable";
  const width = (pct == null ? 0 : Math.min(100, pct)) + "%";
  return html`
    <div class="bar-row">
      <div class="bar-label"><span>${label}</span><span class=${na ? "na" : ""}>${text}</span></div>
      <div class="bar-track"><div class=${fillClass} style=${{ width }}></div></div>
    </div>`;
}

function ScrollPre({ id, text, extraClass }) {
  const shown = text == null ? "unavailable" : text;
  const na = shown === "unavailable";
  return html`<pre id=${id} class="scroll-pre ${extraClass || ""} ${na ? "na" : ""}">${shown}</pre>`;
}

// ---- top bar ----

function ConnBanner({ ok, error }) {
  const hidden = ok !== false;
  return html`
    <div id="connBanner" class="conn-banner" hidden=${hidden}>
      disconnected from labbench server -- retrying${error ? " (" + error + ")" : ""}
    </div>`;
}

function ConnIndicator({ ok }) {
  const cls = ok === null ? "conn-unknown" : ok ? "conn-ok" : "conn-error";
  const text = ok === null ? "connecting" : ok ? "connected" : "disconnected";
  return html`<span id="connIndicator" class="conn-indicator ${cls}"><span class="conn-dot"></span>${text}</span>`;
}

function ConfigSummary({ config }) {
  if (!config) return html`<span id="configSummary" class="config-summary na">unavailable</span>`;
  const bits = [
    fmtOr(config.model), fmtOr(config.quantization), "ctx=" + fmtOr(config.max_model_len),
    "kv=" + fmtOr(config.kv_gib, "GiB"), "spec=" + fmtOr(config.spec),
  ];
  return html`<span id="configSummary" class="config-summary">${bits.join("  |  ")}</span>`;
}

const BACKEND_VALUES = ["vllm", "baseline", "engine"];

function BackendGroup({ backend, onSwitch }) {
  const active = backend ? backend.active : null;
  const status = backend ? backend.status : null;
  const switching = status === "switching";
  return html`
    <fieldset id="backendGroup" class="backend-group segctrl" disabled=${switching}>
      <legend>backend</legend>
      ${BACKEND_VALUES.map((v) => html`
        <label class="segctrl-option" key=${v}>
          <input type="radio" name="backend" value=${v} id=${"backend-" + v}
            checked=${status !== "failed" && active === v}
            disabled=${switching}
            onChange=${() => {}}
            onClick=${(e) => { e.preventDefault(); if (!switching) onSwitch(v); }} />
          <span>${v}</span>
        </label>`)}
    </fieldset>`;
}

function SwitchStatus({ backend }) {
  if (!backend) return html`<span id="switchStatus" class="switch-status"></span>`;
  const { status, stage, elapsed_s, error, active } = backend;
  if (status === "switching") {
    return html`<span id="switchStatus" class="switch-status">switching: ${fmtOr(stage)} (${fmtNum(elapsed_s, 1)}s)</span>`;
  }
  if (status === "failed") {
    return html`<span id="switchStatus" class="switch-status error">failed: ${fmtOr(error)}</span>`;
  }
  if (status === "stopped" || active === null) {
    return html`<span id="switchStatus" class="switch-status">${active === null ? "no backend active" : ""}</span>`;
  }
  return html`<span id="switchStatus" class="switch-status"></span>`;
}

function QuantSelect({ quantization, backend, value, focused, onFocusChange, onCommit }) {
  useEffect(() => {
    if (!focused && quantization != null) onCommit(quantization, /* announce */ false);
  }, [quantization, focused]);
  const switching = backend && backend.status === "switching";
  return html`
    <select id="quantSelect" disabled=${switching} value=${value}
      onFocus=${() => onFocusChange(true)}
      onBlur=${() => onFocusChange(false)}
      onChange=${(e) => onCommit(e.target.value, true)}>
      <option value="bf16">bf16</option>
      <option value="fp8">fp8</option>
      <option value="int4">int4</option>
    </select>`;
}

function ThinkingToggle({ checked, onChange }) {
  return html`
    <label class="switch-toggle thinking-toggle">
      <input type="checkbox" id="thinkingToggle" checked=${checked} onChange=${(e) => onChange(e.target.checked)} />
      <span class="switch-track"><span class="switch-thumb"></span></span>
      <span class="field-inline-text">thinking</span>
    </label>`;
}

function TopBar({ conn, config, backend, quantValue, quantFocused, onQuantFocusChange,
                   onQuantCommit, onBackendSwitch, thinkingEnabled, onThinkingChange }) {
  return html`
    <header id="topbar" class="topbar">
      <div class="topbar-row topbar-row-status">
        <${ConnIndicator} ok=${conn.ok} />
        <${ConfigSummary} config=${config} />
      </div>
      <div class="topbar-row topbar-controls">
        <${BackendGroup} backend=${backend} onSwitch=${onBackendSwitch} />
        <label class="field-inline quant-label">
          <span class="field-inline-text">quant</span>
          <${QuantSelect} quantization=${config ? config.quantization : null} backend=${backend}
            value=${quantValue} focused=${quantFocused} onFocusChange=${onQuantFocusChange} onCommit=${onQuantCommit} />
        </label>
        <${SwitchStatus} backend=${backend} />
        <${ThinkingToggle} checked=${thinkingEnabled} onChange=${onThinkingChange} />
      </div>
    </header>`;
}

// ---- chat pane ----

const MessageRow = memo(function MessageRow({ msg }) {
  const blocks = useMemo(() => (msg.role === "assistant" ? renderMarkdownElements(msg.content) : null), [msg.content, msg.role]);
  return html`
    <div class="msg msg-${msg.role}">
      <div class="msg-role">${msg.role}</div>
      ${msg.role === "assistant" && msg.reasoningContent ? html`
        <details class="thinking-block">
          <summary>thinking (${msg.reasoningContent.length} chars)</summary>
          <div class="thinking-content">${msg.reasoningContent}</div>
        </details>` : null}
      <div class="msg-bubble">${msg.role === "assistant" ? blocks : msg.content}</div>
    </div>`;
});

const MessageList = memo(function MessageList({ messages }) {
  return html`${messages.map((m) => html`<${MessageRow} key=${m.id} msg=${m} />`)}`;
});

// Owns its own state so an arriving token re-renders only this component, never
// the parent App or the instrument panels. getSnapshot() lets the caller read the
// final text without subscribing to it.
const StreamingMessage = memo(forwardRef(function StreamingMessage({ containerRef, pinnedRef }, ref) {
  const [content, setContent] = useState("");
  const [reasoning, setReasoning] = useState("");
  const contentRef = useRef("");
  const reasoningRef = useRef("");

  useImperativeHandle(ref, () => ({
    appendContent(chunk) { contentRef.current += chunk; setContent(contentRef.current); },
    appendReasoning(chunk) { reasoningRef.current += chunk; setReasoning(reasoningRef.current); },
    getSnapshot() { return { content: contentRef.current, reasoning: reasoningRef.current }; },
  }), []);

  useEffect(() => {
    if (pinnedRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  });

  const blocks = useMemo(() => renderMarkdownElements(content), [content]);

  return html`
    <div class="msg msg-assistant">
      <div class="msg-role">assistant</div>
      ${reasoning ? html`
        <details class="thinking-block">
          <summary>thinking (${reasoning.length} chars)</summary>
          <div class="thinking-content">${reasoning}</div>
        </details>` : null}
      <div class="msg-bubble">${blocks}</div>
    </div>`;
}));

function ChatPane({ messages, streamingActive, streamRef, onSend, onStop, inputDisabled, sending }) {
  const containerRef = useRef(null);
  const pinnedRef = useRef(true);
  const inputRef = useRef(null);
  const [showJump, setShowJump] = useState(false);

  const onScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    const pinned = dist < 40;
    pinnedRef.current = pinned;
    setShowJump(!pinned);
  }, []);

  const jumpToLatest = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    pinnedRef.current = true;
    setShowJump(false);
  }, []);

  useEffect(() => {
    if (pinnedRef.current && containerRef.current) containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [messages, streamingActive]);

  const handleSend = () => {
    const text = inputRef.current.value.trim();
    if (!text) return;
    inputRef.current.value = "";
    onSend(text);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return html`
    <section class="chat-pane" id="chatPane">
      <div class="chat-messages" id="chatMessages" ref=${containerRef} onScroll=${onScroll}>
        <${MessageList} messages=${messages} />
        ${streamingActive ? html`<${StreamingMessage} ref=${streamRef} containerRef=${containerRef} pinnedRef=${pinnedRef} />` : null}
      </div>
      <button id="jumpLatest" class="jump-latest" hidden=${!showJump} type="button" onClick=${jumpToLatest}>jump to latest ↓</button>
      <div class="composer">
        <div class="composer-field">
          <textarea id="chatInput" ref=${inputRef} disabled=${inputDisabled}
            placeholder="message... (enter to send, shift+enter for newline)" onKeyDown=${handleKeyDown}></textarea>
          <div class="composer-buttons">
            <button id="sendBtn" class="btn btn-primary" type="button" disabled=${inputDisabled || sending} onClick=${handleSend}>send</button>
            <button id="stopBtn" class="btn btn-stop" type="button" hidden=${!sending} onClick=${onStop}>stop</button>
          </div>
        </div>
      </div>
    </section>`;
}

// ---- instrument panels ----

function RequestPanel({ requestDisplay, clientTtftMs }) {
  const pending = requestDisplay.kind === "pending";
  const trace = pending ? null : requestDisplay.trace;
  const prediction = pending ? null : requestDisplay.prediction;

  let ttftText, ttftNa;
  if (pending) { ttftText = "pending"; ttftNa = true; }
  else if (trace && isNum(trace.ttft_ms)) { ttftText = fmtNum(trace.ttft_ms, 0); ttftNa = false; }
  else { ttftText = "unavailable"; ttftNa = true; }

  const predVal = prediction && prediction.values ? prediction.values.ttft_ms : null;
  let predText, gapText;
  if (!pending && isNum(predVal)) {
    predText = fmtNum(predVal, 0);
    if (trace && isNum(trace.ttft_ms) && predVal !== 0) {
      const gap = ((trace.ttft_ms - predVal) / predVal) * 100;
      gapText = (gap >= 0 ? "+" : "") + gap.toFixed(1) + "%";
    } else {
      gapText = "unavailable";
    }
  } else {
    predText = pending ? "unavailable" : withError("unavailable", prediction && prediction.error);
    gapText = "unavailable";
  }

  let interText;
  if (!pending && trace && (isNum(trace.inter_event_p50_ms) || isNum(trace.inter_event_p95_ms))) {
    interText = fmtNum(trace.inter_event_p50_ms, 1) + " / " + fmtNum(trace.inter_event_p95_ms, 1) + " ms";
  } else {
    interText = "unavailable";
  }

  const predNa = predText === "unavailable" || predText.indexOf("unavailable") === 0;
  const gapNa = gapText === "unavailable";

  return html`
    <section class="panel" id="panel-request">
      <header class="panel-head"><h2>this request</h2></header>
      <div class="panel-body">
        <div class="metric-headline">
          <div class="metric-headline-row">
            <div class="metric-value${ttftNa ? " na" : ""}" id="req-ttft">${ttftText}</div>
            <div class="metric-unit">ms</div>
          </div>
          <div class="metric-label">ttft</div>
          <div class="metric-sub">predicted <span class=${predNa ? "na" : ""} id="req-ttft-pred">${predText}</span> ms · gap <span class=${gapNa ? "na" : ""} id="req-ttft-gap">${gapText}</span></div>
        </div>
        <${MetricRow} label="client-side (includes network)" text=${rowText(clientTtftMs, 0, " ms")} />
        <${MetricRow} label="inter-event latency p50 / p95" text=${interText} />
        <${MetricRow} label="per-token itl" text=${rowText(trace && trace.itl_ms_derived, 1, " ms")} />
        <${MetricRow} label="tokens per event" text=${rowText(trace && trace.tokens_per_event, 2)} />
        <${MetricRow} label="prompt tokens" text=${rowText(trace && trace.prompt_tokens, 0)} />
        <${MetricRow} label="cached tokens" text=${rowText(trace && trace.cached_tokens, 0)} />
        <${MetricRow} label="completion tokens" text=${rowText(trace && trace.completion_tokens, 0)} />
      </div>
    </section>`;
}

function EngineNowPanel({ engine }) {
  if (!engine || engine.error) {
    return html`
      <section class="panel" id="panel-engine">
        <header class="panel-head"><h2>engine now</h2></header>
        <div class="panel-body">
          <${MetricRow} label="running" text="unavailable" />
          <${MetricRow} label="waiting" text="unavailable" />
          <${BarRow} label="kv cache" text="unavailable" pct=${0} fillClass="bar-fill" />
          <${MetricRow} label="prefix cache hit rate" text="unavailable" />
          <details class="bound-details"><summary>metric bindings</summary><${ScrollPre} id="eng-bound" text=${null} extraClass="small-pre" /></details>
          <div class="probe-error" id="eng-error">${engine && engine.error ? engine.error : ""}</div>
        </div>
      </section>`;
  }
  const v = engine.values || {};
  return html`
    <section class="panel" id="panel-engine">
      <header class="panel-head"><h2>engine now</h2></header>
      <div class="panel-body">
        <${MetricRow} label="running" text=${rowText(v.running, 0)} />
        <${MetricRow} label="waiting" text=${rowText(v.waiting, 0)} />
        <${BarRow} label="kv cache" text=${isNum(v.kv_usage) ? fmtPctFrac(v.kv_usage, 1) : "unavailable"}
          pct=${isNum(v.kv_usage) ? v.kv_usage * 100 : 0} fillClass=${isNum(v.kv_usage) ? "bar-fill bar-engine" : "bar-fill"} />
        <${MetricRow} label="prefix cache hit rate" text=${isNum(v.prefix_hit_rate) ? fmtPctFrac(v.prefix_hit_rate, 1) : "unavailable"} />
        <details class="bound-details"><summary>metric bindings</summary><${ScrollPre} id="eng-bound" text=${engine.bound ? JSON.stringify(engine.bound, null, 2) : null} extraClass="small-pre" /></details>
        <div class="probe-error" id="eng-error"></div>
      </div>
    </section>`;
}

function GpuNowPanel({ gpu }) {
  const noData = !gpu || gpu.error || !gpu.devices || !gpu.devices.length;
  if (noData) {
    return html`
      <section class="panel" id="panel-gpu">
        <header class="panel-head"><h2>gpu now</h2></header>
        <div class="panel-body">
          <${BarRow} label="memory" text="unavailable" pct=${0} fillClass="bar-fill bar-gpu-mem" />
          <${BarRow} label="power" text="unavailable" pct=${0} fillClass="bar-fill bar-power" />
          <${MetricRow} label="utilization" text="unavailable" />
          <${MetricRow} label="temperature" text="unavailable" />
          <table class="proc-table" id="gpu-proc-table">
            <thead><tr><th>pid</th><th>used mib</th></tr></thead>
            <tbody id="gpu-proc-body"></tbody>
          </table>
          <div class="probe-error" id="gpu-error">${gpu && gpu.error ? gpu.error : (gpu ? "no devices reported" : "")}</div>
        </div>
      </section>`;
  }
  const d = gpu.devices[0];
  const memOk = isNum(d.memory_used) && isNum(d.memory_total) && d.memory_total > 0;
  const memPct = memOk ? (d.memory_used / d.memory_total) * 100 : 0;
  const memText = memOk ? fmtNum(d.memory_used, 0) + " / " + fmtNum(d.memory_total, 0) + " MiB (" + memPct.toFixed(1) + "%)" : "unavailable";
  const pwrOk = isNum(d.power_draw) && isNum(d.power_limit) && d.power_limit > 0;
  const pwrPct = pwrOk ? (d.power_draw / d.power_limit) * 100 : 0;
  const pwrText = pwrOk ? fmtNum(d.power_draw, 0) + " / " + fmtNum(d.power_limit, 0) + " W" : "unavailable";
  const procs = gpu.processes || [];
  return html`
    <section class="panel" id="panel-gpu">
      <header class="panel-head"><h2>gpu now</h2></header>
      <div class="panel-body">
        <${BarRow} label="memory" text=${memText} pct=${memPct} fillClass="bar-fill bar-gpu-mem" />
        <${BarRow} label="power" text=${pwrText} pct=${pwrPct} fillClass="bar-fill bar-power" />
        <${MetricRow} label="utilization" text=${rowText(d.utilization_gpu, 0, "%")} />
        <${MetricRow} label="temperature" text=${rowText(d.temperature_gpu, 0, " C")} />
        <table class="proc-table" id="gpu-proc-table">
          <thead><tr><th>pid</th><th>used mib</th></tr></thead>
          <tbody id="gpu-proc-body">
            ${procs.length === 0
              ? html`<tr><td colspan="2" class="na">unavailable</td></tr>`
              : procs.map((p, i) => html`<tr key=${i}><td>${fmtOr(p.pid)}</td><td>${fmtNum(p.used_mib, 0)}</td></tr>`)}
          </tbody>
        </table>
        <div class="probe-error" id="gpu-error"></div>
      </div>
    </section>`;
}

function TtftChart({ turnTtfts }) {
  if (!turnTtfts.length) return html`<div class="ttft-chart" id="conv-ttft-chart"></div>`;
  const max = Math.max(1, ...turnTtfts.filter(isNum));
  return html`
    <div class="ttft-chart" id="conv-ttft-chart">
      ${turnTtfts.map((v, i) => isNum(v)
        ? html`<div key=${i} class="ttft-bar" style=${{ height: Math.max(2, (v / max) * 100) + "%" }} title=${v.toFixed(0) + " ms"}></div>`
        : html`<div key=${i} class="ttft-bar na" style=${{ height: "100%" }} title="unavailable"></div>`)}
    </div>`;
}

function ConversationPanel({ turns, promptTokens, maxModelLen, turnTtfts }) {
  const ok = isNum(promptTokens) && isNum(maxModelLen) && maxModelLen > 0;
  const pct = ok ? (promptTokens / maxModelLen) * 100 : 0;
  const text = ok ? promptTokens + " / " + maxModelLen + " (" + pct.toFixed(1) + "%)" : "unavailable";
  const fillClass = ok ? "bar-fill " + (pct > 95 ? "state-error" : pct > 80 ? "state-warning" : "state-normal") : "bar-fill";
  return html`
    <section class="panel" id="panel-conversation">
      <header class="panel-head"><h2>this conversation</h2></header>
      <div class="panel-body">
        <${MetricRow} label="turns" text=${String(turns)} />
        <${BarRow} label="context" text=${text} pct=${pct} fillClass=${fillClass} />
        <div class="ttft-chart-wrap">
          <div class="ttft-chart-title">ttft per turn</div>
          <${TtftChart} turnTtfts=${turnTtfts} />
        </div>
      </div>
    </section>`;
}

function ConfigPanel({ config }) {
  if (!config) {
    return html`
      <section class="panel" id="panel-config">
        <header class="panel-head"><h2>config</h2></header>
        <div class="panel-body">
          <${MetricRow} label="model" text="unavailable" />
          <${MetricRow} label="quantization" text="unavailable" />
          <${MetricRow} label="max model len" text="unavailable" />
          <${MetricRow} label="kv tokens" text="unavailable" />
          <${MetricRow} label="kv gib" text="unavailable" />
          <${MetricRow} label="spec" text="unavailable" />
        </div>
      </section>`;
  }
  return html`
    <section class="panel" id="panel-config">
      <header class="panel-head"><h2>config</h2></header>
      <div class="panel-body">
        <${MetricRow} label="model" text=${fmtOr(config.model)} />
        <${MetricRow} label="quantization" text=${fmtOr(config.quantization)} />
        <${MetricRow} label="max model len" text=${rowText(config.max_model_len, 0)} />
        <${MetricRow} label="kv tokens" text=${rowText(config.kv_tokens, 0)} />
        <${MetricRow} label="kv gib" text=${rowText(config.kv_gib, 2, " GiB")} />
        <${MetricRow} label="spec" text=${fmtOr(config.spec)} />
      </div>
    </section>`;
}

const Instruments = memo(function Instruments({ requestDisplay, clientTtftMs, engine, gpu, turns, promptTokens, maxModelLen, turnTtfts, config }) {
  return html`
    <aside class="instruments" id="instruments">
      <${RequestPanel} requestDisplay=${requestDisplay} clientTtftMs=${clientTtftMs} />
      <${EngineNowPanel} engine=${engine} />
      <${GpuNowPanel} gpu=${gpu} />
      <${ConversationPanel} turns=${turns} promptTokens=${promptTokens} maxModelLen=${maxModelLen} turnTtfts=${turnTtfts} />
      <${ConfigPanel} config=${config} />
    </aside>`;
});

// ---- bottom drawer ----

function PromptTab({ prompt, nTokens, onRefresh }) {
  return html`
    <div class="tab-panel active" id="tab-prompt">
      <div class="tab-toolbar">
        <button id="promptRefresh" class="btn" type="button" onClick=${onRefresh}>refresh</button>
        <span>tokens: <span id="promptNTokens" class="mono-inline">${rowText(nTokens, 0)}</span></span>
      </div>
      <${ScrollPre} id="promptOutput" text=${prompt} />
    </div>`;
}

function WireTab({ trace }) {
  const hasEvents = trace && trace.events;
  const summary = hasEvents
    ? "n_events=" + fmtOr(trace.n_events) + "  n_content_events=" + fmtOr(trace.n_content_events) +
      "  tokens_per_event=" + (isNum(trace.tokens_per_event) ? trace.tokens_per_event.toFixed(2) : "unavailable")
    : "unavailable";
  return html`
    <div class="tab-panel active" id="tab-wire">
      <div class="tab-toolbar"><span id="wireSummary" class=${summary === "unavailable" ? "na" : ""}>${summary}</span></div>
      <div class="scroll-pre table-scroll">
        <table class="wire-table">
          <thead><tr><th>#</th><th>t_ms</th><th>bytes</th><th>chars</th></tr></thead>
          <tbody id="wireBody">
            ${hasEvents ? trace.events.map((ev, i) => html`
              <tr key=${i}><td>${i}</td><td>${fmtOr(ev.t_ms)}</td><td>${fmtOr(ev.bytes)}</td><td>${fmtOr(ev.chars)}</td></tr>`) : null}
          </tbody>
        </table>
      </div>
    </div>`;
}

function MetricsTab({ metricsRaw, filter, onFilterChange, onRefresh }) {
  const text = useMemo(() => {
    if (!filter) return metricsRaw || "unavailable";
    const lower = filter.toLowerCase();
    return metricsRaw.split("\n").filter((l) => l.toLowerCase().includes(lower)).join("\n");
  }, [metricsRaw, filter]);
  return html`
    <div class="tab-panel active" id="tab-metrics">
      <div class="tab-toolbar">
        <input id="metricsFilter" type="text" placeholder="filter (case-insensitive)" value=${filter} onInput=${(e) => onFilterChange(e.target.value)} />
        <button id="metricsRefresh" class="btn" type="button" onClick=${onRefresh}>refresh</button>
      </div>
      <${ScrollPre} id="metricsOutput" text=${text} />
    </div>`;
}

function GpuTab({ gpu }) {
  return html`<div class="tab-panel active" id="tab-gpu"><${ScrollPre} id="gpuJson" text=${gpu ? JSON.stringify(gpu, null, 2) : null} /></div>`;
}

function ServerTab({ backend, config, journalText }) {
  const info = { backend: backend || "unavailable", launch_cmd: config ? fmtOr(config.launch_cmd) : "unavailable" };
  return html`
    <div class="tab-panel active" id="tab-server">
      <${ScrollPre} id="serverInfo" text=${JSON.stringify(info, null, 2)} />
      <${ScrollPre} id="serverJournal" text=${journalText} extraClass="journal-pre" />
    </div>`;
}

function BombardTab({ n, promptTokens, maxTokens, onFieldChange, onRun, running, output }) {
  return html`
    <div class="tab-panel active" id="tab-bombard">
      <div class="tab-toolbar bombard-toolbar">
        <label>n <input id="bombardN" type="number" min="1" value=${n} onChange=${(e) => onFieldChange("n", e.target.value)} /></label>
        <label>prompt tokens <input id="bombardPromptTokens" type="number" min="1" value=${promptTokens} onChange=${(e) => onFieldChange("promptTokens", e.target.value)} /></label>
        <label>max tokens <input id="bombardMaxTokens" type="number" min="1" value=${maxTokens} onChange=${(e) => onFieldChange("maxTokens", e.target.value)} /></label>
        <button id="bombardRun" class="btn btn-primary" type="button" disabled=${running} onClick=${onRun}>run</button>
      </div>
      <${ScrollPre} id="bombardOutput" text=${output} />
    </div>`;
}

const TAB_LABELS = ["prompt", "wire", "metrics", "gpu", "server", "bombard"];

function Drawer({ open, onToggle, activeTab, onTabClick, tabProps }) {
  return html`
    <section class="drawer ${open ? "open" : ""}" id="drawer">
      <div class="drawer-bar">
        <button id="drawerToggle" class="drawer-toggle" type="button" onClick=${onToggle}>
          <span class="drawer-toggle-chevron">▲</span>drawer
        </button>
        <nav class="drawer-tabs" id="drawerTabs">
          ${TAB_LABELS.map((t) => html`
            <button key=${t} class="tab-btn ${activeTab === t ? "active" : ""}" type="button" onClick=${() => onTabClick(t)}>${t}</button>`)}
        </nav>
      </div>
      <div class="drawer-body" id="drawerBody">
        ${activeTab === "prompt" ? html`<${PromptTab} ...${tabProps.prompt} />` : html`<div class="tab-panel" id="tab-prompt"></div>`}
        ${activeTab === "wire" ? html`<${WireTab} ...${tabProps.wire} />` : html`<div class="tab-panel" id="tab-wire"></div>`}
        ${activeTab === "metrics" ? html`<${MetricsTab} ...${tabProps.metrics} />` : html`<div class="tab-panel" id="tab-metrics"></div>`}
        ${activeTab === "gpu" ? html`<${GpuTab} ...${tabProps.gpu} />` : html`<div class="tab-panel" id="tab-gpu"></div>`}
        ${activeTab === "server" ? html`<${ServerTab} ...${tabProps.server} />` : html`<div class="tab-panel" id="tab-server"></div>`}
        ${activeTab === "bombard" ? html`<${BombardTab} ...${tabProps.bombard} />` : html`<div class="tab-panel" id="tab-bombard"></div>`}
      </div>
    </section>`;
}

// ---- root component ----

function App() {
  const [conn, setConn] = useState({ ok: null, error: null });
  const [pollState, setPollState] = useState(null);
  const [requestDisplay, setRequestDisplay] = useState({ kind: "pending" });
  const [finalTrace, setFinalTrace] = useState(MOCK ? mockFixtures().traces.traces[0] : null);
  const [clientTtftMs, setClientTtftMs] = useState(null);
  const [turnTtfts, setTurnTtfts] = useState(MOCK ? [187.3, 203.9, 165.0] : []);
  const [messages, setMessages] = useState(MOCK ? initialMockMessages() : []);
  const [streamingActive, setStreamingActive] = useState(false);
  const [sending, setSending] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(true);
  const [quantValue, setQuantValue] = useState("bf16");
  const [quantFocused, setQuantFocused] = useState(false);
  const [activeTab, setActiveTab] = useState("prompt");
  const [drawerOpen, setDrawerOpen] = useState(false);

  const [promptOutput, setPromptOutput] = useState(null);
  const [promptNTokens, setPromptNTokens] = useState(null);
  const [metricsRaw, setMetricsRaw] = useState("");
  const [metricsFilter, setMetricsFilter] = useState("");
  const [journalText, setJournalText] = useState(null);
  const [bombardFields, setBombardFields] = useState({ n: 8, promptTokens: 512, maxTokens: 64 });
  const [bombardOutput, setBombardOutput] = useState(null);
  const [bombardRunning, setBombardRunning] = useState(false);

  const idCounter = useRef(0);
  const nextId = () => "m" + idCounter.current++;

  const streamHandleRef = useRef(null);
  const abortCtrlRef = useRef(null);
  const streamingActiveRef = useRef(false);
  const pollStateRef = useRef(null);

  useEffect(() => { streamingActiveRef.current = streamingActive; }, [streamingActive]);
  useEffect(() => { pollStateRef.current = pollState; }, [pollState]);

  // ---- 500ms state poll: backend, config, engine, gpu, and the live trace while streaming ----
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const state = await apiGet("/labbench/state");
        if (cancelled) return;
        setPollState(state);
        setConn({ ok: true, error: null });
        if (streamingActiveRef.current) {
          try {
            const data = await apiGet("/labbench/traces?n=1");
            const list = data.traces || [];
            const live = list.length ? list[list.length - 1] : null;
            if (live && live.status === "streaming") setRequestDisplay({ kind: "trace", trace: live, prediction: null });
          } catch (e) { /* the state poll already reports connection loss */ }
        }
      } catch (e) {
        if (!cancelled) setConn({ ok: false, error: e.message });
      } finally {
        if (!cancelled) setTimeout(poll, 500);
      }
    }
    poll();
    return () => { cancelled = true; };
  }, []);

  // ---- server journal: polls forever, cadence tied to switch status ----
  useEffect(() => {
    let cancelled = false;
    async function pollJournal() {
      try {
        const data = await apiGet("/labbench/journal?lines=200");
        if (cancelled) return;
        setJournalText(data.error ? withError("unavailable", data.error) : (data.lines || []).join("\n"));
      } catch (e) {
        if (!cancelled) setJournalText(withError("unavailable", e.message));
      } finally {
        if (!cancelled) {
          const switching = pollStateRef.current && pollStateRef.current.backend && pollStateRef.current.backend.status === "switching";
          setTimeout(pollJournal, switching ? 2000 : 10000);
        }
      }
    }
    pollJournal();
    return () => { cancelled = true; };
  }, []);

  const refreshPrompt = useCallback(async () => {
    try {
      const data = await apiPost("/labbench/render", { messages: buildApiMessages(messages), enable_thinking: thinkingEnabled });
      if (data.error) {
        setPromptOutput(withError("unavailable", data.error));
        setPromptNTokens(null);
      } else {
        setPromptOutput(fmtOr(data.prompt));
        setPromptNTokens(data.n_tokens);
      }
    } catch (e) {
      setPromptOutput(withError("unavailable", e.message));
      setPromptNTokens(null);
    }
  }, [messages, thinkingEnabled]);

  useEffect(() => { refreshPrompt(); }, []); // eslint-disable-line

  const refreshMetrics = useCallback(async () => {
    try {
      const data = await apiGet("/labbench/metrics/raw");
      setMetricsRaw(data.error ? "" : (data.text || ""));
      if (data.error) setMetricsRaw(""); // keep the toolbar error visible via the rendered "unavailable"
    } catch (e) {
      setMetricsRaw("");
    }
  }, []);

  async function finalizeTrace() {
    try {
      const data = await apiGet("/labbench/traces?n=1");
      const trace = data.traces && data.traces.length ? data.traces[data.traces.length - 1] : null;
      setFinalTrace(trace);
      setTurnTtfts((prev) => [...prev, trace && isNum(trace.ttft_ms) ? trace.ttft_ms : null]);
      let prediction = null;
      if (trace && isNum(trace.prompt_tokens)) {
        try { prediction = await apiGet("/labbench/prediction?context=" + trace.prompt_tokens); }
        catch (e) { prediction = { error: e.message, values: {} }; }
      }
      setRequestDisplay({ kind: "trace", trace, prediction });
    } catch (e) {
      setTurnTtfts((prev) => [...prev, null]);
      setRequestDisplay({ kind: "pending" });
    }
  }

  async function handleSend(text) {
    if (sending) return;
    const backend = pollState && pollState.backend;
    if (!backend || backend.status === "switching" || backend.active === null) return;

    const userMsg = { id: nextId(), role: "user", content: text, reasoningContent: "" };
    const updated = [...messages, userMsg];
    setMessages(updated);
    const apiMessages = buildApiMessages(updated);

    setStreamingActive(true);
    setSending(true);
    setRequestDisplay({ kind: "pending" });

    const clientStart = performance.now();
    let firstContentAt = null;

    function applyDelta(delta) {
      const handle = streamHandleRef.current;
      if (!handle) return;
      if (delta.reasoning_content) handle.appendReasoning(delta.reasoning_content);
      if (delta.content) {
        if (firstContentAt === null) {
          firstContentAt = performance.now();
          setClientTtftMs(firstContentAt - clientStart);
        }
        handle.appendContent(delta.content);
      }
    }

    function finalize() {
      const snap = streamHandleRef.current ? streamHandleRef.current.getSnapshot() : { content: "", reasoning: "" };
      const assistantMsg = { id: nextId(), role: "assistant", content: snap.content, reasoningContent: snap.reasoning };
      setMessages((prev) => [...prev, assistantMsg]);
      setStreamingActive(false);
      setSending(false);
      abortCtrlRef.current = null;
      finalizeTrace();
    }

    if (MOCK) {
      await mockStreamChat(applyDelta);
      finalize();
      return;
    }

    const ctrl = new AbortController();
    abortCtrlRef.current = ctrl;
    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          model: (pollState && pollState.config && pollState.config.served_model) || "labbench",
          messages: apiMessages,
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
      const handle = streamHandleRef.current;
      if (handle) handle.appendContent(e.name !== "AbortError" ? "\n\n[error: " + e.message + "]" : "\n\n[stopped]");
    } finally {
      finalize();
    }
  }

  function handleStop() {
    if (abortCtrlRef.current) abortCtrlRef.current.abort();
  }

  async function handleBackendSwitch(v) {
    const quantization = v === "vllm" ? quantValue : null;
    try { await apiPost("/labbench/backend", { backend: v, quantization }); }
    catch (e) { console.error("backend switch failed", e); }
  }

  async function handleQuantCommit(value, announce) {
    setQuantValue(value);
    if (!announce) return;
    const backend = pollState && pollState.backend;
    if (!backend || backend.active !== "vllm" || backend.status === "switching") return;
    try { await apiPost("/labbench/backend", { backend: "vllm", quantization: value }); }
    catch (e) { /* the poll reports the outcome */ }
  }

  function handleTabClick(tab) {
    setActiveTab(tab);
    if (tab === "prompt") refreshPrompt();
    if (tab === "metrics") refreshMetrics();
  }

  function handleBombardField(field, raw) {
    const n = parseInt(raw, 10);
    setBombardFields((prev) => ({ ...prev, [field]: Number.isNaN(n) ? prev[field] : n }));
  }

  async function pollBombard() {
    try {
      const data = await apiGet("/labbench/bombard");
      setBombardOutput(fmtOr(data.stdout));
      if (data.running) {
        setTimeout(pollBombard, 1000);
      } else {
        setBombardRunning(false);
      }
    } catch (e) {
      setBombardOutput(withError("unavailable", e.message));
      setBombardRunning(false);
    }
  }

  async function handleBombardRun() {
    setBombardRunning(true);
    setBombardOutput("starting...");
    try {
      await apiPost("/labbench/bombard", { n: bombardFields.n, prompt_tokens: bombardFields.promptTokens, max_tokens: bombardFields.maxTokens });
      pollBombard();
    } catch (e) {
      setBombardOutput(withError("unavailable", e.message));
      setBombardRunning(false);
    }
  }

  const backend = pollState ? pollState.backend : null;
  const config = pollState ? pollState.config : null;
  const engine = pollState ? pollState.engine : null;
  const gpu = pollState ? pollState.gpu : null;
  const controlsDisabled = !backend || backend.status === "switching" || backend.active === null;
  const assistantTurns = useMemo(() => messages.filter((m) => m.role === "assistant").length, [messages]);
  const promptTokens = finalTrace ? finalTrace.prompt_tokens : null;
  const maxModelLen = config ? config.max_model_len : null;

  const tabProps = {
    prompt: { prompt: promptOutput, nTokens: promptNTokens, onRefresh: refreshPrompt },
    wire: { trace: finalTrace },
    metrics: { metricsRaw, filter: metricsFilter, onFilterChange: setMetricsFilter, onRefresh: refreshMetrics },
    gpu: { gpu },
    server: { backend, config, journalText },
    bombard: {
      n: bombardFields.n, promptTokens: bombardFields.promptTokens, maxTokens: bombardFields.maxTokens,
      onFieldChange: handleBombardField, onRun: handleBombardRun, running: bombardRunning, output: bombardOutput,
    },
  };

  return html`
    <${Fragment}>
      <${ConnBanner} ok=${conn.ok} error=${conn.error} />
      <${TopBar} conn=${conn} config=${config} backend=${backend}
        quantValue=${quantValue} quantFocused=${quantFocused} onQuantFocusChange=${setQuantFocused}
        onQuantCommit=${handleQuantCommit} onBackendSwitch=${handleBackendSwitch}
        thinkingEnabled=${thinkingEnabled} onThinkingChange=${setThinkingEnabled} />
      <main class="main">
        <${ChatPane} messages=${messages} streamingActive=${streamingActive} streamRef=${streamHandleRef}
          onSend=${handleSend} onStop=${handleStop} inputDisabled=${controlsDisabled} sending=${sending} />
        <${Instruments} requestDisplay=${requestDisplay} clientTtftMs=${clientTtftMs} engine=${engine} gpu=${gpu}
          turns=${assistantTurns} promptTokens=${promptTokens} maxModelLen=${maxModelLen} turnTtfts=${turnTtfts} config=${config} />
      </main>
      <${Drawer} open=${drawerOpen} onToggle=${() => setDrawerOpen((o) => !o)} activeTab=${activeTab} onTabClick=${handleTabClick} tabProps=${tabProps} />
    </${Fragment}>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);

// ==== MOCK MODE (?mock=1) ====================================================
// Every fixture the UI can consume lives in this one function. apiGet/apiPost
// route here instead of fetch() when MOCK is true, and handleSend() calls
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

function initialMockMessages() {
  return [
    { id: "mock-u1", role: "user", content: "what is speculative decoding", reasoningContent: "" },
    {
      id: "mock-a1", role: "assistant",
      content: "Speculative decoding drafts several tokens with a small model and verifies them in one batch with the large model.",
      reasoningContent: "The user is asking for a short definition; keep it to one sentence.",
    },
  ];
}

})();
