import { useCallback, useEffect, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Activity, ArrowDown, CircleStop, Copy, Moon, PanelBottom, Play, Send, Sun } from "lucide-react";
import { BenchTurn, LiveTurn, type LiveTurnHandle } from "./components/ai-elements/message";
import { TraceChart } from "./components/TraceChart";
import { get, isNumber, post, unavailable, valueText } from "./lib";
import type { BenchState, Journal, Prediction, RenderedPrompt, Trace, Turn } from "./types";

type Theme = "cream" | "dark";
type TraceEnvelope = { traces?: Trace[]; upstream?: string };
type RawMetrics = { text?: string; error?: string | null };
type Bombard = { job?: string | null; running?: boolean; stdout?: string; returncode?: number | null; error?: string | null };
type TabName = "request" | "prompt" | "wire" | "metrics" | "gpu" | "server" | "predictions" | "bombard";

function formatUnavailable(value: unknown, error?: string | null, unit = "", digits = 1, percent = false) {
  if (typeof value === "string" && value.trim()) return value;
  if (!isNumber(value)) return unavailable(error);
  return valueText(percent ? value * 100 : value, digits, percent ? "%" : unit);
}

function Metric({ label, value, error, unit = "", digits = 1, note, percent = false }: {
  label: string; value: unknown; error?: string | null; unit?: string; digits?: number;
  note?: string; percent?: boolean;
}) {
  const formatted = formatUnavailable(value, error, unit, digits, percent);
  const available = isNumber(value) || (typeof value === "string" && value.trim().length > 0);
  return <div className="metric-row">
    <span className="metric-label">{label}{note && <small>{note}</small>}</span>
    <strong className={available ? undefined : "metric-unavailable"} title={formatted}>{formatted}</strong>
  </div>;
}

function Panel({ title, eyebrow, children, className = "" }: {
  title: string; eyebrow?: string; children: ReactNode; className?: string;
}) {
  return <section className={`instrument-panel ${className}`}>
    <div className="panel-heading"><div>{eyebrow && <span>{eyebrow}</span>}<h2>{title}</h2></div></div>
    {children}
  </section>;
}

function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return <button className="copy-button" type="button" title={done ? "Copied" : label}
    aria-label={done ? "Copied" : label} onClick={() => {
      void navigator.clipboard.writeText(text).then(() => {
        setDone(true); window.setTimeout(() => setDone(false), 1200);
      }).catch(() => {});
    }}><Copy size={13} />{done ? "Copied" : label}</button>;
}

export function App() {
  const [theme, setTheme] = useState<Theme>(() => document.documentElement.dataset.theme === "dark" ? "dark" : "cream");
  const [benchState, setBenchState] = useState<BenchState | null>(null);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [traceEnvelope, setTraceEnvelope] = useState<TraceEnvelope | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [thinkingEnabled, setThinkingEnabled] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabName>("request");
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [requestAt, setRequestAt] = useState<string | null>(null);
  const [requestJson, setRequestJson] = useState("");
  const [requestMessages, setRequestMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [rendered, setRendered] = useState<RenderedPrompt | null>(null);
  const [previousPrompt, setPreviousPrompt] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [journal, setJournal] = useState<Journal | null>(null);
  const [rawMetrics, setRawMetrics] = useState<RawMetrics | null>(null);
  const [bombard, setBombard] = useState<Bombard | null>(null);
  const [bombardFields, setBombardFields] = useState({ n: "8", prompt_tokens: "512", max_tokens: "64" });
  const [atBottom, setAtBottom] = useState(true);
  const [toast, setToast] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const abortRef = useRef<AbortController | null>(null);
  const liveTurnRef = useRef<LiveTurnHandle | null>(null);
  const lastPromptRef = useRef<string | null>(null);

  const pollNow = useCallback(async () => {
    const [stateData, traceData] = await Promise.all([
      get<BenchState>("/labbench/state"), get<TraceEnvelope>("/labbench/traces?n=24"),
    ]);
    setBenchState(stateData);
    setTraces(traceData.traces ?? []);
    setTraceEnvelope(traceData);
    setConnectionError(null);
  }, []);

  useEffect(() => {
    let active = true;
    let timer = 0;
    const poll = async () => {
      try { await pollNow(); }
      catch (reason) { if (active) setConnectionError(reason instanceof Error ? reason.message : "Probe request failed"); }
      if (active) timer = window.setTimeout(poll, 500);
    };
    void poll();
    return () => { active = false; window.clearTimeout(timer); };
  }, [pollNow]);

  useEffect(() => {
    let active = true;
    let timer = 0;
    const poll = async () => {
      try { const value = await get<Journal>("/labbench/journal?lines=200"); if (active) setJournal(value); }
      catch (reason) { if (active) setJournal({ error: reason instanceof Error ? reason.message : "Request failed" }); }
      if (active) timer = window.setTimeout(poll, benchState?.backend?.status === "switching" ? 2000 : 10000);
    };
    void poll();
    return () => { active = false; window.clearTimeout(timer); };
  }, [benchState?.backend?.status]);

  useEffect(() => {
    let active = true;
    get<RawMetrics>("/labbench/metrics/raw").then((value) => { if (active) setRawMetrics(value); })
      .catch((reason: unknown) => { if (active) setRawMetrics({ error: reason instanceof Error ? reason.message : "Request failed" }); });
    return () => { active = false; };
  }, []);

  const toggleTheme = () => {
    const next: Theme = theme === "cream" ? "dark" : "cream";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("llm-ui-theme", next); } catch {}
  };

  const jumpBottom = () => {
    if (!scrollRef.current) return;
    pinnedRef.current = true;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    setAtBottom(true);
  };

  useEffect(() => {
    if (pinnedRef.current && scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, sending]);

  const refreshPrompt = async (apiMessages: Array<{ role: string; content: string }>) => {
    setRequestMessages(apiMessages);
    try {
      const result = await post<RenderedPrompt>("/labbench/render", {
        messages: apiMessages, enable_thinking: thinkingEnabled,
      });
      if (result.error) { setRendered(result); return; }
      if (result.prompt != null) {
        setPreviousPrompt(lastPromptRef.current);
        lastPromptRef.current = result.prompt;
      }
      setRendered(result);
    } catch (reason) {
      setRendered({ prompt: "", error: reason instanceof Error ? reason.message : "Prompt render failed" });
    }
  };

  const handleSend = async () => {
    const text = draft.trim();
    if (!text || sending) return;
    const nextMessages: Turn[] = [...messages, { id: Date.now(), role: "user", content: text, reasoning: "" }];
    const apiMessages = nextMessages.map(({ role, content }) => ({ role, content }));
    const model = benchState?.config?.served_model || benchState?.config?.model || "labbench";
    const body = {
      model, messages: apiMessages, stream: true, stream_options: { include_usage: true },
      max_tokens: 1024, temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0,
      chat_template_kwargs: { enable_thinking: thinkingEnabled },
    };
    const exactBody = JSON.stringify(body);
    setMessages(nextMessages);
    setDraft("");
    setError(null);
    setSending(true);
    pinnedRef.current = true;
    setRequestAt(new Date().toISOString());
    setRequestJson(exactBody);
    setPrediction(null);
    void refreshPrompt(apiMessages);
    let streamedContent = "";
    let streamedReasoning = "";
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST", headers: { "content-type": "application/json" },
        body: exactBody, signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error(`Upstream request failed (${response.status})`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (raw === "[DONE]") continue;
          let event: Record<string, unknown>;
          try { event = JSON.parse(raw) as Record<string, unknown>; } catch { continue; }
          const choice = Array.isArray(event.choices) ? event.choices[0] as { delta?: Record<string, unknown> } | undefined : undefined;
          const delta = choice?.delta;
          if (!delta) continue;
          const thought = typeof delta.reasoning === "string" ? delta.reasoning
            : typeof delta.reasoning_content === "string" ? delta.reasoning_content : "";
          const content = typeof delta.content === "string" ? delta.content : "";
          if (thought || content) {
            streamedContent += content;
            streamedReasoning += thought;
            if (thought) liveTurnRef.current?.appendReasoning(thought);
            if (content) liveTurnRef.current?.appendContent(content);
          }
        }
      }
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "The request failed.");
    } finally {
      abortRef.current = null;
      if (streamedContent || streamedReasoning || controller.signal.aborted) {
        setMessages((before) => [...before, {
          id: Date.now(), role: "assistant", content: streamedContent,
          reasoning: streamedReasoning, stopped: controller.signal.aborted,
        }]);
      }
      setSending(false);
      try {
        const envelope = await get<TraceEnvelope>("/labbench/traces?n=1");
        const latest = envelope.traces?.at(-1) ?? null;
        if (latest) {
          setTraceEnvelope(envelope);
          setPrediction(isNumber(latest.prompt_tokens)
            ? await get<Prediction>(`/labbench/prediction?context=${latest.prompt_tokens}`).catch((reason: unknown) => ({ error: reason instanceof Error ? reason.message : "Prediction failed" }))
            : { error: "request prompt token count unavailable", values: {} });
        }
        await pollNow();
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Could not read the completed trace.");
      }
    }
  };

  const stop = () => abortRef.current?.abort();
  const changeBackend = async (backend: string) => {
    const quantization = backend === "vllm" ? benchState?.config?.quantization ?? "bf16" : null;
    try {
      const result = await post<{ error?: string | null }>("/labbench/backend", { backend, quantization });
      if (result.error) setToast(result.error);
      else setToast(`Requested ${backend}; waiting for server state.`);
    } catch (reason) { setToast(reason instanceof Error ? reason.message : "Backend request failed"); }
    window.setTimeout(() => setToast(null), 3500);
  };
  const changeQuant = async (event: ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value;
    try {
      const result = await post<{ error?: string | null }>("/labbench/backend", { backend: "vllm", quantization: value });
      setToast(result.error || "Quantization change requested; waiting for server state.");
    } catch (reason) { setToast(reason instanceof Error ? reason.message : "Quantization request failed"); }
    window.setTimeout(() => setToast(null), 3500);
  };

  const startBombard = async () => {
    try {
      const result = await post<Bombard>("/labbench/bombard", {
        n: Number(bombardFields.n), prompt_tokens: Number(bombardFields.prompt_tokens),
        max_tokens: Number(bombardFields.max_tokens),
      });
      setBombard(result);
      if (result.error) return;
      let active = true;
      const poll = async () => {
        if (!active) return;
        try {
          const status = await get<Bombard>("/labbench/bombard");
          setBombard(status);
          if (status.running) window.setTimeout(poll, 900);
        } catch (reason) { setBombard({ error: reason instanceof Error ? reason.message : "Bombard status unavailable" }); }
      };
      void poll();
      window.setTimeout(() => { active = false; }, 10 * 60 * 1000);
    } catch (reason) { setBombard({ error: reason instanceof Error ? reason.message : "Bombard request failed" }); }
  };

  const activeTrace = traces.at(-1) ?? null;
  const config = benchState?.config;
  const backend = benchState?.backend;
  const engine = benchState?.engine;
  const gpu = benchState?.gpu;
  const device = gpu?.devices?.[0];
  const gpuReason = gpu?.error ?? (device ? null : "nvidia-smi returned no GPU device");
  const engineValues = engine?.values ?? {};
  const contextWidth = isNumber(activeTrace?.prompt_tokens) && isNumber(config?.max_model_len) && config.max_model_len > 0
    ? `${Math.min(100, activeTrace.prompt_tokens / config.max_model_len * 100)}%` : "0%";
  const promptText = rendered?.prompt ?? "";
  const deltaOffset = previousPrompt == null ? 0 : (() => {
    const old = previousPrompt;
    const next = promptText;
    let index = 0;
    while (index < old.length && index < next.length && old[index] === next[index]) index++;
    return index;
  })();
  const promptDelta = promptText.slice(deltaOffset);

  const tabs: Array<[TabName, string]> = [
    ["request", "Request"], ["prompt", "Prompt"], ["wire", "Wire"], ["metrics", "Metrics"],
    ["gpu", "GPU"], ["server", "Server"], ["predictions", "Predictions"], ["bombard", "Bombard"],
  ];

  return <div className="bench-shell">
    <header className="bench-header">
      <div className="bench-brand"><div className="bench-mark"><Activity size={17} /></div>
        <div><strong>Inference</strong><span>LAB BENCH</span></div></div>
      <div className="config-strip" aria-label="Inference configuration">
        <div><span>Model</span><b>{config?.served_model || config?.model || unavailable(config?.served_model_error)}</b></div>
        <div><span>Weights</span><b>{config?.quantization || unavailable()}</b></div>
        <div><span>KV dtype</span><b>{config?.kv_dtype || unavailable("not exposed by frozen API")}</b></div>
        <div><span>Speculation</span><b>{config?.spec == null ? (config ? "off" : unavailable()) : config.spec}</b></div>
        <div><span>Max context</span><b>{formatUnavailable(config?.max_model_len)}</b></div>
        <div><span>KV budget</span><b>{config?.kv_tokens != null && config?.kv_gib != null
          ? `${valueText(config.kv_tokens, 0)} tokens / ${valueText(config.kv_gib, 2)} GiB`
          : unavailable()}</b></div>
      </div>
      <div className="header-tools">
        <span className={`connection-state ${connectionError ? "disconnected" : "connected"}`}>
          <i />{connectionError ? "PROBES OFFLINE" : "LIVE 2 HZ"}
        </span>
        <button className="icon-button" type="button" onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "cream" : "dark"} theme`} title="Switch theme">
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>

    <div className="bench-toolbar">
      <div className="backend-choice" role="radiogroup" aria-label="Inference backend">
        {[["baseline", "Baseline"], ["engine", "Engine"], ["vllm", "vLLM"]].map(([value, label]) =>
          <label key={value} className={backend?.active === value ? "backend-selected" : ""}>
            <input type="radio" name="backend" value={value} checked={backend?.active === value}
              onChange={() => void changeBackend(value)} disabled={backend?.status === "switching"} />{label}
          </label>)}
      </div>
      <div className="quant-control"><span>Quantization</span>
        <select value={config?.quantization ?? ""} disabled={backend?.active !== "vllm" || backend.status === "switching"}
          onChange={(event) => void changeQuant(event)} aria-label="vLLM quantization">
          <option value="bf16">BF16</option><option value="fp8">FP8</option><option value="int4">INT4</option>
        </select>
      </div>
      <label className="thinking-toggle"><input type="checkbox" checked={thinkingEnabled}
        onChange={(event) => setThinkingEnabled(event.target.checked)} /><span>Thinking</span></label>
      <div className="backend-status" role="status">
        <span className={`status-dot status-${backend?.status ?? "unknown"}`} />
        {backend?.status ?? "State unavailable"}{backend?.stage ? ` / ${backend.stage}` : ""}
        {isNumber(backend?.elapsed_s) && backend.elapsed_s > 0 ? ` / ${valueText(backend.elapsed_s, 1, "s")}` : ""}
        {backend?.error ? <span className="status-error">{backend.error}</span> : null}
      </div>
    </div>

    {connectionError && <div className="bench-alert" role="status">Probe connection unavailable: {connectionError}</div>}
    {error && <div className="bench-alert bench-alert-error" role="alert">{error}<button onClick={() => setError(null)} aria-label="Dismiss error">×</button></div>}

    <main className="bench-workspace">
      <section className="bench-chat-column" aria-label="Bench chat">
        <div className="chat-column-header"><div><span>INFERENCE SESSION</span><h1>Live request</h1></div>
          <span className="turn-count">{messages.filter((message) => message.role === "assistant").length} completed turns</span></div>
        <div className="bench-chat-scroll" ref={scrollRef} onScroll={(event) => {
          const element = event.currentTarget;
          const near = element.scrollHeight - element.scrollTop - element.clientHeight < 55;
          pinnedRef.current = near; setAtBottom(near);
        }}>
          <div className="bench-chat-body">
            {!messages.length && !sending && <div className="bench-empty">
              <div className="empty-rule" /><h2>Send a request to inspect the full path.</h2>
              <p>Prompt assembly, the engine wire and live hardware probes appear alongside each turn.</p>
            </div>}
            <AnimatePresence initial={false}>
              {messages.map((message) => <motion.div key={message.id} initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.14 }}>
                <BenchTurn turn={message} theme={theme} />
              </motion.div>)}
            </AnimatePresence>
            {sending && <LiveTurn ref={liveTurnRef} theme={theme} scrollRef={scrollRef} pinnedRef={pinnedRef} />}
          </div>
        </div>
        {!atBottom && <button className="chat-jump" type="button" onClick={jumpBottom} aria-label="Jump to latest"><ArrowDown size={15} /></button>}
        <form className="bench-composer" onSubmit={(event) => { event.preventDefault(); void handleSend(); }}>
          <textarea value={draft} rows={1} aria-label="Message" placeholder="Ask the model"
            disabled={sending} onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault(); void handleSend();
              }
            }} />
          {sending ? <button className="send-control stop-control" type="button" onClick={stop} aria-label="Stop generation"><CircleStop size={16} />Stop</button>
            : <button className="send-control" type="submit" disabled={!draft.trim()}><Send size={15} />Send</button>}
        </form>
      </section>

      <aside className="instrument-column" aria-label="Live instruments">
        <div className="instrument-scroll">
          <Panel title="This request" eyebrow="MEASURED">
            <div className="headline-metric"><div><span>TTFT</span><strong>{formatUnavailable(activeTrace?.ttft_ms, activeTrace?.error, " ms", 1)}</strong></div>
              <div className="prediction-compare"><small>Roofline</small><b>{formatUnavailable(prediction?.values?.ttft_ms, prediction?.error, " ms", 1)}</b></div>
              <div className="prediction-compare gap-value"><small>Gap</small><b>{unavailable("not exposed by frozen API")}</b></div>
            </div>
            <div className="metric-list two-col">
              <Metric label="Engine TTFT" value={null} error="not exposed by labbench trace" unit=" ms" />
              <Metric label="ITL p50" value={activeTrace?.inter_event_p50_ms} error={activeTrace?.error} unit=" ms" />
              <Metric label="ITL p95" value={activeTrace?.inter_event_p95_ms} error={activeTrace?.error} unit=" ms" />
              <Metric label="End to end" value={activeTrace?.e2e_ms} error={activeTrace?.error} unit=" ms" />
            </div>
            <div className="metric-list three-col">
              <Metric label="Prompt" value={activeTrace?.prompt_tokens} error={activeTrace?.error} digits={0} />
              <Metric label="Cached" value={activeTrace?.cached_tokens} error={activeTrace?.error} digits={0} />
              <Metric label="Output" value={activeTrace?.completion_tokens} error={activeTrace?.error} digits={0} />
            </div>
            <div className="ttft-chart-panel"><div className="micro-heading">TTFT / PROMPT BY TURN</div>
              <TraceChart traces={traces} theme={theme} /></div>
          </Panel>

          <Panel title="Engine now" eyebrow="PROMETHEUS">
            <div className="metric-list two-col">
              <Metric label="Running" value={engineValues.running} error={engine?.error} digits={0} />
              <Metric label="Waiting" value={engineValues.waiting} error={engine?.error} digits={0} />
              <Metric label="Prefix cache" value={engineValues.prefix_hit_rate} error={engine?.error} digits={1} percent />
              <Metric label="KV utilization" value={engineValues.kv_usage} error={engine?.error} digits={1} percent />
              <Metric label="Tokens / chunk" value={activeTrace?.tokens_per_event} error={activeTrace?.error} digits={2} />
              <Metric label="Chunk tokens" value={null} error="per-event tokens are not exposed by frozen API" />
            </div>
            <div className="meter-track" aria-label="KV cache utilization">
              <span style={{ width: isNumber(engineValues.kv_usage) ? `${Math.max(0, Math.min(100, engineValues.kv_usage * 100))}%` : "0%" }} />
            </div>
          </Panel>

          <Panel title="GPU now" eyebrow="NVIDIA-SMI">
            <div className="metric-list two-col">
              <Metric label="Memory" value={device?.memory_used} error={gpuReason} unit=" MiB" digits={0} />
              <Metric label="Total memory" value={device?.memory_total} error={gpuReason} unit=" MiB" digits={0} />
              <Metric label="Utilization" value={device?.utilization_gpu} error={gpuReason} unit="%" digits={0} />
              <Metric label="Temperature" value={device?.temperature_gpu} error={gpuReason} unit=" C" digits={0} />
              <Metric label="Power" value={device?.power_draw} error={gpuReason} unit=" W" digits={1} />
              <Metric label="TDP" value={device?.power_limit} error={gpuReason} unit=" W" digits={1} />
            </div>
            <div className="metric-note">Power draw is shown against the device-reported power limit.</div>
          </Panel>

          <Panel title="Context & cache" eyebrow="PROMPT PATH">
            <div className="context-label"><span>Prompt tokens</span><strong>{formatUnavailable(activeTrace?.prompt_tokens, activeTrace?.error, "")}
              <i> / </i>{formatUnavailable(config?.max_model_len)}</strong></div>
            <div className="meter-track context-track"><span style={{ width: contextWidth }} /></div>
            <div className="metric-list two-col">
              <Metric label="KV token budget" value={config?.kv_tokens} digits={0} />
              <Metric label="KV memory" value={config?.kv_gib} unit=" GiB" digits={2} />
              <Metric label="Cached tokens" value={activeTrace?.cached_tokens} error={activeTrace?.error} digits={0} />
              <Metric label="Prompt delta" value={null} error="text delta is in the Prompt drawer" />
            </div>
          </Panel>

          <Panel title="Process" eyebrow="SERVICE">
            <div className="metric-list two-col">
              <Metric label="Active unit" value={backend?.active ?? null} error={backend?.active == null ? backend?.error ?? "no backend active" : null} digits={0} />
              <Metric label="System state" value={backend?.status} error={connectionError} digits={0} />
              <Metric label="Process memory" value={gpu?.processes?.[0]?.used_mib}
                error={gpu?.processes_error ?? (gpu?.processes?.length ? gpuReason : "no compute process rows returned")} unit=" MiB" digits={0} />
              <Metric label="Spec decode" value={engineValues.spec_decode_acceptance} error={engine?.error ?? "metric not exposed by current scrape"} unit="%" digits={1} />
            </div>
          </Panel>
        </div>
      </aside>

      <section className={`deep-drawer ${drawerOpen ? "drawer-open" : "drawer-closed"}`}>
        <div className="drawer-head">
          <div className="drawer-tabs" role="tablist" aria-label="Request detail layers">
            {tabs.map(([name, label]) => <button type="button" key={name} role="tab"
              aria-selected={tab === name} className={tab === name ? "active" : ""}
              onClick={() => { setTab(name); setDrawerOpen(true); }}>{label}</button>)}
          </div>
          <button type="button" className="drawer-toggle" onClick={() => setDrawerOpen((value) => !value)}
            aria-label={drawerOpen ? "Collapse details" : "Expand details"} title={drawerOpen ? "Collapse details" : "Expand details"}>
            <PanelBottom size={15} />
          </button>
        </div>
        {drawerOpen && <div className="drawer-content" role="tabpanel">
          {tab === "request" && <div className="drawer-grid request-detail">
            <Panel title="Browser to service" eyebrow="LAYER 1">
              <div className="metric-row"><span className="metric-label">Client timestamp</span><strong>{requestAt ?? unavailable("no request sent")}</strong></div>
              <div className="metric-row"><span className="metric-label">Upstream proxy</span><strong>{traceEnvelope?.upstream ?? unavailable("trace not received")}</strong></div>
              <div className="detail-toolbar"><span>Exact JSON body sent by browser</span><CopyButton text={requestJson || ""} /></div>
              <pre className="code-view">{requestJson || unavailable("send a request to capture it")}</pre>
            </Panel>
            <Panel title="Request result" eyebrow="TRACE">
              <div className="metric-list two-col">
                <Metric label="Request status" value={activeTrace?.status} error={activeTrace?.error} digits={0} />
                <Metric label="Request id" value={activeTrace?.request_id} error={activeTrace?.error} digits={0} />
                <Metric label="TTFT" value={activeTrace?.ttft_ms} error={activeTrace?.error} unit=" ms" />
                <Metric label="E2E" value={activeTrace?.e2e_ms} error={activeTrace?.error} unit=" ms" />
              </div>
              <div className="detail-note">The browser timestamp is client-side UX context. Latencies above are from the labbench server trace.</div>
            </Panel>
          </div>}
          {tab === "prompt" && <div className="drawer-grid">
            <Panel title="Messages and template" eyebrow="LAYER 2">
              <div className="detail-toolbar"><span>Thinking {thinkingEnabled ? "kept" : "disabled"} in rendered template</span>
                <span>Model tokenizer tokens: {formatUnavailable(rendered?.n_tokens, rendered?.error, "", 0)}</span></div>
              <pre className="code-view prompt-messages">{requestMessages.length ? JSON.stringify(requestMessages, null, 2) : unavailable("no request sent")}</pre>
            </Panel>
            <Panel title="Rendered prompt" eyebrow="PREFIX DELTA">
              {rendered?.error ? <div className="unavailable-note">{unavailable(rendered.error)}</div> : rendered?.prompt == null
                ? <div className="unavailable-note">{unavailable("no request sent")}</div>
                : <><div className="detail-toolbar"><span>{previousPrompt == null ? "First rendered prompt" : "New suffix from previous turn"}</span><span>Cached tokens: {formatUnavailable(activeTrace?.cached_tokens, activeTrace?.error, "", 0)}</span></div>
                  <pre className="code-view prompt-delta"><span>{promptText.slice(0, deltaOffset)}</span><mark>{promptDelta}</mark></pre>
                  <details className="full-prompt"><summary>Full rendered prompt</summary><pre className="code-view">{promptText}</pre></details></>}
            </Panel>
          </div>}
          {tab === "wire" && <div className="drawer-grid">
            <Panel title="Upstream SSE events" eyebrow="LAYER 3">
              <div className="detail-toolbar"><span>{formatUnavailable(activeTrace?.n_content_events, activeTrace?.error, " content events", 0)}</span>
                <span>Per-event token count: {unavailable("not exposed by frozen API")}</span></div>
              <div className="event-table-wrap"><table className="event-table"><thead><tr><th>Arrival (t+)</th><th>Bytes</th><th>Characters</th><th>Tokens</th></tr></thead>
                <tbody>{activeTrace?.events?.length ? activeTrace.events.map((event, index) => <tr key={`${index}-${event.t_ms}`}>
                  <td>{formatUnavailable(event.t_ms, activeTrace.error, " ms", 2)}</td>
                  <td>{formatUnavailable(event.bytes, activeTrace.error, " B", 0)}</td>
                  <td>{formatUnavailable(event.chars, activeTrace.error, "", 0)}</td>
                  <td>{unavailable("not exposed by frozen API")}</td>
                </tr>) : <tr><td colSpan={4}>{unavailable(activeTrace?.error ?? "no events recorded")}</td></tr>}</tbody></table></div>
            </Panel>
            <Panel title="Usage chunk" eyebrow="ENGINE REPORTED">
              <div className="metric-list two-col">
                <Metric label="Prompt tokens" value={activeTrace?.prompt_tokens} error={activeTrace?.error} digits={0} />
                <Metric label="Cached prompt" value={activeTrace?.cached_tokens} error={activeTrace?.error} digits={0} />
                <Metric label="Completion tokens" value={activeTrace?.completion_tokens} error={activeTrace?.error} digits={0} />
                <Metric label="Tokens per event" value={activeTrace?.tokens_per_event} error={activeTrace?.error} digits={2} />
              </div>
              <div className="detail-note">Tokens per event remains an aggregate trace field; it is not merged into event rows.</div>
            </Panel>
          </div>}
          {tab === "metrics" && <div className="drawer-grid">
            <Panel title="Bound engine metrics" eyebrow="LAYER 4">
              {engine?.error ? <div className="unavailable-note">{unavailable(engine.error)}</div>
                : engine?.values && Object.keys(engine.values).length ? <>
                  <div className="metric-list two-col">
                    {Object.entries(engine.values).map(([name, value]) => <Metric key={name} label={name}
                      value={value} error={engine.error ?? "not returned by current scrape"}
                      percent={name === "kv_usage" || name === "prefix_hit_rate" || name === "spec_decode_acceptance"} />)}
                  </div>
                  <details className="full-prompt"><summary>Metric bindings</summary>
                    <pre className="code-view">{JSON.stringify(engine.bound ?? {}, null, 2)}</pre>
                  </details>
                </> : <div className="unavailable-note">{unavailable("no engine metrics returned")}</div>}
            </Panel>
            <Panel title="Raw Prometheus scrape" eyebrow="SOURCE">
              {rawMetrics?.error ? <div className="unavailable-note">{unavailable(rawMetrics.error)}</div>
                : <pre className="code-view raw-metrics">{rawMetrics?.text || unavailable("empty scrape")}</pre>}
            </Panel>
          </div>}
          {tab === "gpu" && <div className="drawer-grid">
            <Panel title="Device and processes" eyebrow="LAYER 5">
              {gpu?.error ? <div className="unavailable-note">{unavailable(gpu.error)}</div>
                : gpu?.devices?.length ? <><pre className="code-view">{JSON.stringify(gpu.devices, null, 2)}</pre>
                  {gpu.processes_error && <div className="unavailable-note">Process table: {unavailable(gpu.processes_error)}</div>}
                  <pre className="code-view">{JSON.stringify(gpu.processes ?? [], null, 2)}</pre></>
                  : <div className="unavailable-note">{unavailable(gpuReason)}</div>}
            </Panel>
            <Panel title="Power against device limit" eyebrow="DYNAMIC BEHAVIOR">
              <div className="power-pair"><div><span>Draw</span><strong>{formatUnavailable(device?.power_draw, gpuReason, " W")}</strong></div>
                <div><span>Limit</span><strong>{formatUnavailable(device?.power_limit, gpuReason, " W")}</strong></div></div>
              <div className="metric-note">Observed readings are direct from nvidia-smi; the client does not derive a utilization percentage.</div>
            </Panel>
          </div>}
          {tab === "server" && <div className="drawer-grid">
            <Panel title="Unit and launch" eyebrow="LAYER 6">
              <div className="metric-list two-col">
                <Metric label="System state" value={backend?.status} error={connectionError} digits={0} />
                <Metric label="Active backend" value={backend?.active} error={backend?.error ?? (backend?.active == null ? "no backend active" : null)} digits={0} />
                <Metric label="Resolved model" value={config?.model} error={config?.served_model_error} digits={0} />
                <Metric label="Unit" value={journal?.unit} error={journal?.error} digits={0} />
              </div>
              <pre className="code-view">{config?.launch_cmd || unavailable(backend?.error ?? "no backend launch command")}</pre>
            </Panel>
            <Panel title="Journal tail" eyebrow="SYSTEMD">
              {journal?.error ? <div className="unavailable-note">{unavailable(journal.error)}</div>
                : <pre className="code-view journal-view">{journal?.lines?.length ? journal.lines.join("\n") : unavailable("no journal lines")}</pre>}
            </Panel>
          </div>}
          {tab === "predictions" && <div className="drawer-grid">
            <Panel title="Roofline prediction" eyebrow="LAYER 7">
              <div className="metric-list two-col">
                <Metric label="Predicted TTFT" value={prediction?.values?.ttft_ms} error={prediction?.error} unit=" ms" />
                <Metric label="Measured TTFT" value={activeTrace?.ttft_ms} error={activeTrace?.error} unit=" ms" />
                <Metric label="Prediction gap" value={null} error="not exposed by frozen API; UI does not calculate it" />
                <Metric label="Prompt context" value={activeTrace?.prompt_tokens} error={activeTrace?.error} digits={0} />
              </div>
              {prediction?.note && <div className="detail-note">{prediction.note}</div>}
              {prediction?.text && <pre className="code-view">{prediction.text}</pre>}
            </Panel>
            <Panel title="Per-turn measurements" eyebrow="MEASURED TRACE">
              <TraceChart traces={traces} theme={theme} />
              <div className="detail-note">The plot uses server trace TTFT and prompt-token fields. It does not infer the gap.</div>
            </Panel>
          </div>}
          {tab === "bombard" && <div className="drawer-grid">
            <Panel title="Load generation" eyebrow="TOOLS/BENCH.PY">
              <div className="bombard-controls">
                {([["n", "Requests"], ["prompt_tokens", "Prompt tokens"], ["max_tokens", "Max output"]] as const).map(([key, label]) =>
                  <label key={key}>{label}<input type="number" min="1" value={bombardFields[key]}
                    onChange={(event) => setBombardFields((current) => ({ ...current, [key]: event.target.value }))} /></label>)}
                <button className="primary-control" type="button" disabled={Boolean(bombard?.running)} onClick={() => void startBombard()}>
                  <Play size={14} />{bombard?.running ? "Running" : "Run"}
                </button>
              </div>
              {bombard?.error && <div className="unavailable-note">{unavailable(bombard.error)}</div>}
            </Panel>
            <Panel title="Harness output" eyebrow="SERVER SIDE">
              <pre className="code-view journal-view">{bombard?.stdout || unavailable("no bombard run")}</pre>
              <div className="detail-note">Load is generated by the server-side benchmark harness, not by the browser.</div>
            </Panel>
          </div>}
        </div>}
      </section>
    </main>
    <AnimatePresence>{toast && <motion.div className="bench-toast" initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} role="status">{toast}</motion.div>}</AnimatePresence>
  </div>;
}

export default App;
