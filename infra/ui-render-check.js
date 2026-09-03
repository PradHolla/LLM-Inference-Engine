// Execute the instrument components with real React and representative props, including
// all-null props. Catches what `node --check` cannot: a deleted reference, a bad property
// access on a wrapper object, a crash when a probe returns null.
const fs = require("fs"), vm = require("vm"), path = require("path");
const UI = path.join(__dirname, "..", "labbench", "ui");

const sandbox = {
  console, module: undefined, exports: undefined, setTimeout, clearTimeout,
  performance: { now: () => 0 }, TextDecoder: class { decode() { return ""; } },
  navigator: {}, location: { search: "" },
  document: { getElementById: () => null, addEventListener() {}, querySelectorAll: () => [] },
  fetch: () => Promise.reject(new Error("no network in the check")),
  URLSearchParams, URL, AbortController, Promise, JSON, Math, Date, Object, Array,
  String, Number, Boolean, Error, Map, Set, isNaN, parseInt, parseFloat, RegExp,
};
sandbox.window = sandbox; sandbox.globalThis = sandbox; sandbox.self = sandbox;
vm.createContext(sandbox);

for (const f of ["vendor/react.production.min.js", "vendor/htm.js"]) {
  vm.runInContext(fs.readFileSync(path.join(UI, f), "utf8"), sandbox, { filename: f });
}
// Load app.js whole. Truncating it left unbalanced braces; instead ReactDOM is stubbed
// so the mount at the bottom is inert while every definition still evaluates.
sandbox.__LABBENCH_TEST__ = {};
sandbox.ReactDOM = { createRoot: () => ({ render() {}, unmount() {} }), render() {} };
vm.runInContext(fs.readFileSync(path.join(UI, "app.js"), "utf8"), sandbox, { filename: "app.js" });

const R = sandbox.React;
const trace = {
  status: "ok", ttft_ms: 48.2, e2e_ms: 3712, itl_ms_derived: 18.9,
  inter_event_p50_ms: 18.9, inter_event_p95_ms: 19.4, tokens_per_event: 1.01,
  prompt_tokens: 283, cached_tokens: null, completion_tokens: 196, n_content_events: 194,
};
const gpu = { devices: [{ memory_used: 22055, memory_total: 23028, utilization_gpu: 0,
                          power_draw: 66, power_limit: 300, temperature_gpu: 39 }],
              processes: [{ pid: 2167, used_mib: 22046 }], error: null };
const engine = { values: { running: 0, waiting: 0, kv_usage: 0.0, prefix_hit_rate: 0.908 },
                 bound: {}, error: null };
const config = { model: "Qwen/Qwen3-8B", quantization: "fp8", max_model_len: 16384,
                 kv_tokens: 74880, kv_gib: 10.28, spec: null };

const cases = [
  ["RequestPanel, populated", "RequestPanel", { requestDisplay: { kind: "trace", trace, prediction: { values: { ttft_ms: 93 } } }, clientTtftMs: 120 }],
  ["RequestPanel, pending",   "RequestPanel", { requestDisplay: { kind: "pending" }, clientTtftMs: null }],
  ["RequestPanel, empty",     "RequestPanel", {}],
  ["EngineNowPanel",          "EngineNowPanel", { engine }],
  ["EngineNowPanel, error",   "EngineNowPanel", { engine: { values: {}, error: "scrape failed" } }],
  ["GpuNowPanel",             "GpuNowPanel", { gpu }],
  ["GpuNowPanel, no gpu",     "GpuNowPanel", { gpu: { devices: [], processes: [], error: "nvidia-smi missing" } }],
  ["ConversationPanel",       "ConversationPanel", { turns: 5, contextTokens: 490, maxModelLen: 16384, turnTtfts: [46, 107, 109, 112, 114], perTurnTokens: 98 }],
  ["ConversationPanel, empty","ConversationPanel", { turns: 0, contextTokens: null, maxModelLen: null, turnTtfts: [], perTurnTokens: 0 }],
  ["ConfigPanel",             "ConfigPanel", { config }],
  ["ConfigPanel, null",       "ConfigPanel", { config: null }],
  ["TtftChart",               "TtftChart", { turnTtfts: [46, 107, 109] }],
  ["TtftChart, empty",        "TtftChart", { turnTtfts: [] }],
];

function walk(node, depth) {
  if (depth > 60 || node == null || typeof node !== "object") return;
  const kids = node.props && node.props.children;
  const list = Array.isArray(kids) ? kids : kids != null ? [kids] : [];
  for (const k of list) {
    if (k && typeof k === "object" && typeof k.type === "function") {
      walk(k.type(k.props || {}), depth + 1);   // force nested components to execute
    } else { walk(k, depth + 1); }
  }
}

let bad = 0;
for (const [name, fn, props] of cases) {
  const C = sandbox.__LABBENCH_TEST__[fn];
  if (typeof C !== "function") { console.log(`  FAIL ${name}: ${fn} is not defined`); bad++; continue; }
  try { walk(C(props), 0); console.log(`  ok   ${name}`); }
  catch (e) { console.log(`  FAIL ${name}: ${e.message}`); bad++; }
}
console.log(bad ? `  RENDER CHECK FAILED (${bad})` : "  RENDER CHECK PASSED");
process.exit(bad ? 1 : 0);
