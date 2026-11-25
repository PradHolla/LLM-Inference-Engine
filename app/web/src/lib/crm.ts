// CRM agent client: typed API calls, the SSE reader and pure view helpers.
// The event types mirror crm/events.py field for field; change them together.
import { request } from "./api";

export type CrmMode = "staff" | "customer";
export type CrmAgent = "service" | "general";
export type AgentInfo = {
  id: CrmAgent; label: string; description: string; protocol: string;
  jobs: { task_type: string; label: string }[];
};
export const DEFAULT_AGENT: CrmAgent = "service";
export const AGENT_LABELS: Record<CrmAgent, string> = { service: "Service desk agent", general: "General CRM agent" };
/** Used when /api/crm/agents cannot be loaded, so the picker still works. */
export const FALLBACK_AGENTS: AgentInfo[] = (Object.keys(AGENT_LABELS) as CrmAgent[]).map((id) => ({
  id, label: AGENT_LABELS[id], description: "", protocol: "", jobs: [],
}));
const AGENT_KEY = "crm-agent";

/** The remembered agent choice; falls back to the service desk agent when storage is empty or blocked. */
export function loadAgent(): CrmAgent {
  try {
    const value = window.localStorage.getItem(AGENT_KEY);
    return value === "service" || value === "general" ? value : DEFAULT_AGENT;
  } catch { return DEFAULT_AGENT; }
}

export function saveAgent(agent: CrmAgent): void {
  try { window.localStorage.setItem(AGENT_KEY, agent); } catch { /* storage unavailable */ }
}

export type RunContext = { today: string | null; focus_ids: string[]; policy_title: string | null };
export type RunExample = { id: string; task_type: string; skill: string };

export type RunStartEvent = {
  type: "run_start"; run_id: string; question: string; model: string; mode: CrmMode; agent?: CrmAgent;
  context: RunContext; example: RunExample | null;
};
export type StepStartEvent = { type: "step_start"; step: number };
export type ThinkingEvent = { type: "thinking"; step: number; text: string };
export type ActionEvent = {
  type: "action"; step: number; call_id: string; name: string; args: Record<string, unknown>;
};
export type ObservationEvent = {
  type: "observation"; step: number; call_id: string; ok: boolean; rows_total: number; rows_shown: number;
  truncated: boolean; columns: string[]; preview: Record<string, unknown>[]; error: string | null;
  elapsed_ms: number; text?: string | null;
};
export type AnswerEvent = {
  type: "answer"; text: string; expected: unknown[] | null; correct: boolean | null;
  reason?: string | null; sources?: { id: string; title: string }[];
};
export type StatsEvent = {
  type: "stats"; steps: number; actions: number; wall_ms: number; prompt_tokens: number;
  cached_tokens: number; completion_tokens: number; est_cost_usd: number;
};
export type ErrorEvent = { type: "error"; message: string };
export type DoneEvent = { type: "done" };

export type CrmEvent = RunStartEvent | StepStartEvent | ThinkingEvent | ActionEvent | ObservationEvent
  | AnswerEvent | StatsEvent | ErrorEvent | DoneEvent;

export const CRM_EVENT_TYPES = ["run_start", "step_start", "thinking", "action", "observation", "answer",
  "stats", "error", "done"] as const;

export type CrmExample = {
  id: string; title: string; question: string; task_type: string; skill: string; today: string | null;
  context_summary: string; mode: CrmMode; has_expected: boolean; job?: string;
};
export type CrmModelOption = { id: string; label: string; note: string };
export type CrmModels = { current: string; switching: boolean; options: CrmModelOption[]; picker?: boolean };
export type AskRequest = {
  question: string | null; example_id: string | null; mode: CrmMode; thinking: boolean; agent: CrmAgent;
};

export const getCrmAgents = () => request<AgentInfo[]>("/api/crm/agents");
export const getCrmExamples = (agent: CrmAgent) =>
  request<CrmExample[]>(`/api/crm/examples?agent=${encodeURIComponent(agent)}`);
export const getCrmModels = () => request<CrmModels>("/api/crm/models");
export const setCrmModel = (id: string) => request<CrmModels>("/api/crm/model", {
  method: "POST", body: JSON.stringify({ id }),
});

function parseLine(line: string): CrmEvent | null {
  if (!line.startsWith("data:")) return null;
  try {
    const event = JSON.parse(line.slice(5).trim()) as { type?: unknown };
    return event && (CRM_EVENT_TYPES as readonly unknown[]).includes(event.type) ? event as CrmEvent : null;
  } catch { return null; }
}

/** Yield typed events from an SSE body, buffering lines that arrive split across chunks. */
export async function* readCrmEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<CrmEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const event = parseLine(line);
        if (event) yield event;
      }
    }
    buffer += decoder.decode();
    const last = parseLine(buffer.trim());
    if (last) yield last;
  } finally {
    try { await reader.cancel(); } catch { /* already closed */ }
  }
}

/** POST a question and stream the run's events; throws on HTTP errors and when the signal aborts. */
export async function* askCrm(body: AskRequest, signal?: AbortSignal): AsyncGenerator<CrmEvent> {
  const response = await fetch("/api/crm/ask", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { const payload = await response.json(); detail = payload.detail || payload.error || detail; } catch { /* not JSON */ }
    throw new Error(String(detail));
  }
  if (!response.body) throw new Error("The server returned an empty stream.");
  yield* readCrmEvents(response.body);
}

export type CallView = { action: ActionEvent; observation: ObservationEvent | null };
export type StepView = { step: number; thinking: string; calls: CallView[] };
export type RunView = {
  start: RunStartEvent | null; steps: StepView[]; answer: AnswerEvent | null; stats: StatsEvent | null;
  errors: string[]; done: boolean;
};

/** Fold a run's events into steps (thinking plus calls, each call paired with its observation). */
export function buildRun(events: CrmEvent[]): RunView {
  const view: RunView = { start: null, steps: [], answer: null, stats: null, errors: [], done: false };
  const stepFor = (index: number): StepView => {
    let step = view.steps.find((item) => item.step === index);
    if (!step) {
      step = { step: index, thinking: "", calls: [] };
      view.steps.push(step);
      view.steps.sort((a, b) => a.step - b.step);
    }
    return step;
  };
  for (const event of events) {
    if (event.type === "run_start") view.start = event;
    else if (event.type === "step_start") stepFor(event.step);
    else if (event.type === "thinking") stepFor(event.step).thinking += event.text;
    else if (event.type === "action") stepFor(event.step).calls.push({ action: event, observation: null });
    else if (event.type === "observation") {
      const call = stepFor(event.step).calls.find((item) => item.action.call_id === event.call_id);
      if (call) call.observation = event;
    } else if (event.type === "answer") view.answer = event;
    else if (event.type === "stats") view.stats = event;
    else if (event.type === "error") view.errors.push(event.message);
    else if (event.type === "done") view.done = true;
  }
  return view;
}

export const ACTION_LABELS: Record<string, string> = {
  summarize_records: "Summarize records", find_by_name: "Find by name", query_records: "Query records",
  get_record: "Get record", search_text: "Search text", search: "Search text", run_sql: "Run SQL",
  respond: "Respond",
  get_case: "Open the case", search_knowledge: "Search the knowledge base", get_article: "Read an article",
  list_issues: "List known issues", list_products: "List products", agent_expertise: "Rank agents by experience",
  agent_workload: "Check agent workload",
};

export function humanize(name: string): string {
  const text = name.replace(/__c$/, "").replace(/[_\s]+/g, " ").trim();
  return text ? text[0].toUpperCase() + text.slice(1) : name;
}

export function actionLabel(name: string): string {
  return ACTION_LABELS[name] ?? humanize(name);
}

const ID_RE = /^[0-9A-Za-z]{15}(?:[0-9A-Za-z]{3})?$/;

/** A Salesforce Id: 15 or 18 alphanumerics with at least one digit and one letter. */
export function isSalesforceId(value: unknown): value is string {
  return typeof value === "string" && ID_RE.test(value) && /\d/.test(value) && /[A-Za-z]/.test(value);
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export function formatInt(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "n/a" : Math.round(value).toLocaleString("en-US");
}

export function formatCost(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "n/a" : `$${value.toFixed(4)}`;
}

export function formatSeconds(ms: number | null | undefined): string {
  return ms == null || !Number.isFinite(ms) ? "n/a" : `${(ms / 1000).toFixed(1)} s`;
}

export type FilterView = { field: string; op: string; value: string };
export type ArgItem =
  | { label: string; kind: "text" | "mono" | "code"; text: string }
  | { label: string; kind: "chips"; chips: string[] }
  | { label: string; kind: "filters"; filters: FilterView[] };

const OP_LABELS: Record<string, string> = { is_null: "is null", not_null: "is not null", not_in: "not in" };

/** The text of one filter chip, such as `Field__c = Owner Assignment`. */
export function filterText(filter: FilterView): string {
  const op = OP_LABELS[filter.op] ?? filter.op;
  return filter.value === "" ? `${filter.field} ${op}` : `${filter.field} ${op} ${filter.value}`;
}

function asFilters(value: unknown): FilterView[] | null {
  if (!Array.isArray(value)) return null;
  const out: FilterView[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") return null;
    const row = item as Record<string, unknown>;
    if (typeof row.field !== "string" || typeof row.op !== "string") return null;
    out.push({ field: row.field, op: row.op, value: row.value == null ? "" : String(row.value) });
  }
  return out;
}

function metricText(args: Record<string, unknown>): string {
  const metric = String(args.metric ?? "count");
  const field = args.field ? String(args.field) : "";
  const field2 = args.field2 ? String(args.field2) : "";
  if (metric === "avg_days_between") return `average days from ${field} to ${field2}`;
  if (metric === "sum_product") return `sum of ${field} x ${field2}`;
  return metric === "count" || !field ? metric : `${metric}(${field})`;
}

/** Turn an action's arguments into readable items; the raw JSON stays available separately. */
export function formatArgs(name: string, rawArgs: Record<string, unknown>): ArgItem[] {
  const args = rawArgs && typeof rawArgs === "object" ? rawArgs : {};
  if (name === "respond") return [];
  if (name === "run_sql") return typeof args.query === "string" ? [{ label: "Query", kind: "code", text: args.query }] : [];
  const items: ArgItem[] = [];
  const summarize = name === "summarize_records";
  const skip = new Set(summarize ? ["field", "field2", "having_value"]
    : name === "search_knowledge" ? ["limit"] : name === "agent_expertise" ? ["top"] : []);
  for (const [key, value] of Object.entries(args)) {
    if (skip.has(key) || value === null || value === undefined || value === "") continue;
    if (summarize && key === "metric") { items.push({ label: "Metric", kind: "text", text: metricText(args) }); continue; }
    if (summarize && key === "having_op") {
      if (value !== "none") items.push({ label: "Keep groups where", kind: "text", text: `metric ${value} ${args.having_value ?? 0}` });
      continue;
    }
    const label = humanize(key);
    const filters = key === "filters" ? asFilters(value) : null;
    if (filters) { if (filters.length) items.push({ label, kind: "filters", filters }); continue; }
    if (Array.isArray(value)) {
      if (value.length) items.push({ label, kind: "chips", chips: value.map((item) => formatCell(item)) });
    } else if (typeof value === "boolean") {
      if (value) items.push({ label, kind: "text", text: "yes" });
    } else if (typeof value === "object") {
      items.push({ label, kind: "code", text: JSON.stringify(value) });
    } else {
      items.push({ label, kind: isSalesforceId(value) ? "mono" : "text", text: String(value) });
    }
  }
  return items;
}
