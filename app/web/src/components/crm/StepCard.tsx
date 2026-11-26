import { useState } from "react";
import { motion } from "motion/react";
import { BookOpen, ChevronDown, CircleAlert, CircleCheck, Code, Database, FileText, FolderOpen, Gauge, List,
  LoaderCircle, Package, Send, ShieldCheck, Sigma, Signpost, Table2, TextSearch, Trophy, UserSearch, type LucideIcon } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import { actionLabel, filterText, formatArgs, formatInt, type ArgItem, type CallView, type StepView } from "../../lib/crm";
import { formatMs } from "../../lib/utils";
import { ResultTable } from "./ResultTable";

const ICONS: Record<string, LucideIcon> = {
  summarize_records: Sigma, find_by_name: UserSearch, query_records: Table2, get_record: FileText,
  search_text: TextSearch, search: TextSearch, run_sql: Database, respond: Send,
  get_case: FolderOpen, search_knowledge: TextSearch, get_article: BookOpen, list_issues: List,
  list_products: Package, agent_expertise: Trophy, agent_workload: Gauge, get_issue_policy: ShieldCheck, check_offers: ShieldCheck, get_my_cases: FolderOpen, route_case: Trophy, choose_job: Signpost,
};

function ArgRow({ item }: { item: ArgItem }) {
  let body;
  if (item.kind === "filters") {
    body = <span className="crm-chips">{item.filters.map((filter, index) =>
      <span key={index} className="crm-chip filter">{filterText(filter)}</span>)}</span>;
  } else if (item.kind === "chips") {
    body = <span className="crm-chips">{item.chips.map((chip, index) =>
      <span key={index} className="crm-chip">{chip}</span>)}</span>;
  } else if (item.kind === "code") {
    body = <pre className="crm-code">{item.text}</pre>;
  } else {
    body = <span className={item.kind === "mono" ? "crm-id" : "crm-arg-text"}>{item.text}</span>;
  }
  return <div className="crm-arg"><dt>{item.label}</dt><dd>{body}</dd></div>;
}

function Observation({ call, live }: { call: CallView; live: boolean }) {
  const obs = call.observation;
  if (!obs && !live) return <div className="crm-obs pending"><span>No result received</span></div>;
  if (!obs) {
    return <div className="crm-obs pending"><LoaderCircle className="crm-spin" size={14} aria-hidden="true" />
      <span>Waiting for the result</span></div>;
  }
  if (obs.ok && obs.text) return <DocumentResult text={obs.text} truncated={obs.truncated} ms={obs.elapsed_ms} />;
  const partial = obs.rows_shown < obs.rows_total;
  return <div className={`crm-obs ${obs.ok ? "ok" : "error"}`}>
    <div className="crm-obs-head">
      {obs.ok ? <CircleCheck size={14} aria-hidden="true" /> : <CircleAlert size={14} aria-hidden="true" />}
      <strong>{obs.ok ? "Result" : "Failed"}</strong>
      {obs.ok && <span>{obs.rows_total === 1 ? "1 row" : `${formatInt(obs.rows_total)} rows`}</span>}
      {obs.ok && partial && <span className="crm-faint">showing {obs.rows_shown} of {formatInt(obs.rows_total)}</span>}
      {obs.ok && obs.truncated && <span className="crm-faint">truncated</span>}
      <span className="crm-obs-time">{formatMs(obs.elapsed_ms)}</span>
    </div>
    {!obs.ok && <p className="crm-obs-error">{obs.error ?? "The action failed."}</p>}
    {obs.ok && obs.rows_total === 0 && <p className="crm-obs-empty">No matching rows.</p>}
    {obs.ok && <ResultTable columns={obs.columns} rows={obs.preview} />}
  </div>;
}

/** An article or a case comes back as text: show its first lines, the rest one click away. */
function DocumentResult({ text, truncated, ms }: { text: string; truncated: boolean; ms: number | null }) {
  const [open, setOpen] = useState(false);
  const lines = text.split("\n");
  const title = lines[0] ?? "";
  const long = text.length > 600;
  return <div className="crm-obs ok">
    <div className="crm-obs-head">
      <CircleCheck size={14} aria-hidden="true" />
      <strong>Result</strong>
      <span className="crm-doc-title">{title}</span>
      {truncated && <span className="crm-faint">truncated</span>}
      <span className="crm-obs-time">{formatMs(ms)}</span>
    </div>
    <pre className={`crm-doc ${open || !long ? "open" : ""}`}>{lines.slice(1).join("\n")}</pre>
    {long && <button type="button" className="crm-doc-toggle" onClick={() => setOpen((v) => !v)}>
      {open ? "Show less" : "Show full text"}</button>}
  </div>;
}

function ActionCard({ call, live }: { call: CallView; live: boolean }) {
  const { action } = call;
  const [raw, setRaw] = useState(false);
  const Icon = ICONS[action.name] ?? Code;
  const items = formatArgs(action.name, action.args);
  const final = action.name === "respond";
  return <div className={`crm-call ${final ? "final" : ""}`}>
    <div className="crm-call-head">
      <span className="crm-call-icon"><Icon size={15} aria-hidden="true" /></span>
      <strong>{actionLabel(action.name)}</strong>
      <button type="button" className="crm-raw" aria-pressed={raw} onClick={() => setRaw((value) => !value)}>
        {raw ? "Readable" : "Raw"}
      </button>
    </div>
    {raw ? <pre className="crm-code raw">{JSON.stringify(action.args, null, 2)}</pre>
      : items.length > 0 && <dl className="crm-args">{items.map((item, index) => <ArgRow key={index} item={item} />)}</dl>}
    {final ? <p className="crm-call-note">Sent the final answer.</p> : <Observation call={call} live={live} />}
  </div>;
}

function Thinking({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return <Collapsible open={open} onOpenChange={setOpen} className="crm-thinking">
    <CollapsibleTrigger className="crm-thinking-trigger">
      <span className="thinking-mark" aria-hidden="true" />
      <span>Thinking</span>
      <ChevronDown className={open ? "reasoning-chevron open" : "reasoning-chevron"} size={14} />
    </CollapsibleTrigger>
    <CollapsibleContent><div className="reasoning-copy">{text}</div></CollapsibleContent>
  </Collapsible>;
}

/** One model step: a numbered rail marker, collapsed thinking, and an action card per call. */
export function StepCard({ step, live, last }: { step: StepView; live: boolean; last: boolean }) {
  const waiting = live && last && step.calls.length === 0;
  return <motion.section className="crm-step" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.22, ease: "easeOut" }} aria-label={`Step ${step.step + 1}`}>
    <div className="crm-step-marker" aria-hidden="true"><span>{step.step + 1}</span></div>
    <div className="crm-step-body">
      {step.thinking && <Thinking text={step.thinking} />}
      {step.calls.map((call) => <ActionCard key={call.action.call_id} call={call} live={live} />)}
      {waiting && <div className="crm-step-wait" role="status"><span className="pipeline-dot" />Deciding the next action</div>}
    </div>
  </motion.section>;
}
