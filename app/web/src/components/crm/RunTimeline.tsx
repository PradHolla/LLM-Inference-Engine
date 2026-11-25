import { Fragment, useEffect, useState, type ReactNode } from "react";
import { motion } from "motion/react";
import { BookOpen, Bot, Calendar, Check, CircleAlert, Cpu, Lock, Sparkles, TriangleAlert, UserRound, X } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";
import { AGENT_LABELS, buildRun, formatCell, formatCost, formatInt, formatSeconds, humanize, isSalesforceId,
  type AnswerEvent, type CrmAgent, type CrmEvent, type CrmMode, type StatsEvent } from "../../lib/crm";
import { StepCard } from "./StepCard";

const WORD_RE = /\b[0-9A-Za-z]{15,18}\b/g;

/** Text with any Salesforce Ids set in monospace. */
function WithIds({ text }: { text: string }) {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(WORD_RE)) {
    if (!isSalesforceId(match[0])) continue;
    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(<span key={match.index} className="crm-id">{match[0]}</span>);
    last = match.index + match[0].length;
  }
  parts.push(text.slice(last));
  return <>{parts.map((part, index) => <Fragment key={index}>{part}</Fragment>)}</>;
}

function Badge({ answer }: { answer: AnswerEvent }) {
  if (answer.correct === true) {
    return <span className="crm-badge good"><Check size={13} aria-hidden="true" />Matches the expected answer</span>;
  }
  if (answer.correct === false) {
    const expected = (answer.expected ?? []).map((item) => formatCell(item)).join(", ");
    return <span className="crm-badge bad"><X size={13} aria-hidden="true" />
      Does not match{expected ? ` (expected: ${expected})` : ""}</span>;
  }
  return null;
}

function AnswerCard({ answer }: { answer: AnswerEvent }) {
  return <motion.section className="crm-answer" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.26, ease: "easeOut" }} aria-label="Answer">
    <div className="crm-answer-head"><Sparkles size={14} aria-hidden="true" /><span>Answer</span><Badge answer={answer} /></div>
    <p className="crm-answer-text"><WithIds text={answer.text} /></p>
    {answer.reason && <p className="crm-answer-reason"><span>Why</span><WithIds text={answer.reason} /></p>}
    {answer.sources && answer.sources.length > 0 &&
      <p className="crm-answer-sources"><span>Based on</span>
        {answer.sources.map((s) => <span key={s.id} className="crm-chip" title={s.id}>{s.title || s.id}</span>)}</p>}
    {answer.correct === null && answer.expected && answer.expected.length > 0 &&
      <p className="crm-answer-expected"><span>Expected answer</span>
        {answer.expected.map((item) => formatCell(item)).join(", ")}</p>}
  </motion.section>;
}

function Stat({ label, children }: { label: string; children: ReactNode }) {
  return <div className="crm-stat"><dt>{label}</dt><dd>{children}</dd></div>;
}

function StatsFooter({ stats }: { stats: StatsEvent }) {
  const tip = "GPU time on a $1.21/hr A10G";
  return <motion.dl className="crm-stats" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
    <Stat label="Steps">{stats.steps}</Stat>
    <Stat label="Actions">{stats.actions}</Stat>
    <Stat label="Time">{formatSeconds(stats.wall_ms)}</Stat>
    <Stat label="Tokens (prompt / cached / completion)">
      {formatInt(stats.prompt_tokens)} / {formatInt(stats.cached_tokens)} / {formatInt(stats.completion_tokens)}
    </Stat>
    <Stat label="Estimated cost">
      <Tooltip>
        <TooltipTrigger><span className="crm-cost" tabIndex={0} aria-label={`${formatCost(stats.est_cost_usd)}. ${tip}`}>
          {formatCost(stats.est_cost_usd)}</span></TooltipTrigger>
        <TooltipContent>{tip}</TooltipContent>
      </Tooltip>
    </Stat>
  </motion.dl>;
}

function Working({ step }: { step: number }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return <div className="crm-working" role="status">
    <span className="thinking-pulse" aria-hidden="true" />
    <span>Working{step > 0 ? `, step ${step}` : ""}</span>
    <span className="crm-faint">{seconds} s</span>
  </div>;
}

const MODE_LABEL: Record<CrmMode, string> = { staff: "Staff", customer: "Customer" };

/** Render one run live from its events: question header, step cards, answer, stats and errors. */
export function RunTimeline({ events, running, stopped = false, question, mode, agent }: {
  events: CrmEvent[]; running: boolean; stopped?: boolean; question?: string; mode?: CrmMode; agent?: CrmAgent;
}) {
  const run = buildRun(events);
  const start = run.start;
  const shownMode = start?.mode ?? mode;
  const shownAgent = start?.agent ?? agent;
  const context = start?.context;
  const finished = run.answer !== null || run.done || run.errors.length > 0;
  return <article className="crm-run">
    <header className="crm-question">
      <p className="crm-kicker">Question</p>
      <h2>{start?.question ?? question ?? ""}</h2>
      <div className="crm-chip-row">
        {shownAgent && <span className="crm-chip meta agent"><Bot size={12} aria-hidden="true" />
          {AGENT_LABELS[shownAgent] ?? humanize(shownAgent)}</span>}
        {context?.today && <span className="crm-chip meta"><Calendar size={12} aria-hidden="true" />Today {context.today}</span>}
        {shownMode && <span className="crm-chip meta"><UserRound size={12} aria-hidden="true" />{MODE_LABEL[shownMode]} mode</span>}
        {context?.customer && <span className="crm-chip meta"><Lock size={12} aria-hidden="true" />
          Logged in as {context.customer}: sees only their own records</span>}
        {context?.policy_title && <span className="crm-chip meta"><BookOpen size={12} aria-hidden="true" />{context.policy_title}</span>}
        {start?.model && <span className="crm-chip meta"><Cpu size={12} aria-hidden="true" />{start.model}</span>}
        {context?.focus_ids.map((id) => <span key={id} className="crm-chip meta">
          Focus <span className="crm-id">{id}</span></span>)}
        {start?.example && <span className="crm-chip meta">{humanize(start.example.task_type)}</span>}
      </div>
    </header>
    <div className="crm-steps">
      {run.steps.map((step, index) => <StepCard key={step.step} step={step}
        live={running && !finished} last={index === run.steps.length - 1} />)}
    </div>
    {running && !finished && (run.steps.length === 0 || run.steps[run.steps.length - 1].calls.length > 0)
      && <Working step={run.steps.length} />}
    {run.answer && <AnswerCard answer={run.answer} />}
    {run.errors.map((message, index) => <motion.div key={index} className="crm-error" role="alert"
      initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
      <TriangleAlert size={16} aria-hidden="true" />
      <div><strong>The run did not complete</strong><p>{message}</p></div>
    </motion.div>)}
    {stopped && <p className="crm-stopped"><CircleAlert size={13} aria-hidden="true" />Stopped before the run finished.</p>}
    {run.stats && <StatsFooter stats={run.stats} />}
  </article>;
}
