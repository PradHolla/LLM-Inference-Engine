import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Check, ChevronDown, CircleAlert, Lock, ShieldAlert, ShieldCheck, TriangleAlert, X } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import { activityLine, buildRun, formatCell, formatCost, formatInt, formatSeconds, liveStatus, sourceTitle,
  type AnswerEvent, type AnswerFlag, type CrmEvent, type CrmMode, type RunView, type StatsEvent } from "../../lib/crm";
import { StepCard } from "./StepCard";

function Badge({ answer }: { answer: AnswerEvent }) {
  if (answer.correct === true) {
    return <span className="crm-badge good"><Check size={13} aria-hidden="true" />Matches the expected answer</span>;
  }
  if (answer.correct === false) {
    const expected = (answer.expected ?? []).map((item) => formatCell(item)).join(", ");
    return <span className="crm-badge bad"><X size={13} aria-hidden="true" />
      Expected {expected || "a different answer"}</span>;
  }
  return null;
}

/** A calm notice that case text addressed to the agent was found; the agent flags it and carries on. */
function PlantedNotice({ flags }: { flags: AnswerFlag[] }) {
  return <div className="crm-flag" role="note">
    <p className="crm-flag-head"><ShieldAlert size={15} aria-hidden="true" />
      <span><strong>Needs review</strong> This case contains text addressed to the agent.</span></p>
    {flags.slice(0, 3).map((flag, index) => <blockquote key={index} className="crm-flag-item">
      <p>{`\u201C${flag.sentence}\u201D`}</p>
      <cite>from {flag.source}</cite>
    </blockquote>)}
  </div>;
}

/** The answer as a person reads it: the answer itself, then the checked source it came from. */
function AnswerCard({ answer }: { answer: AnswerEvent }) {
  const text = answer.display?.text ?? answer.text;
  const cites = answer.citations ?? [];
  const flags = (answer.flags ?? []).filter((flag) => flag.sentence);
  return <motion.section className="crm-answer" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.26, ease: "easeOut" }} aria-label="Answer">
    <p className="crm-answer-text">{text}</p>
    {flags.length > 0 && <PlantedNotice flags={flags} />}
    {answer.display?.detail && <p className="crm-answer-detail">{answer.display.detail}</p>}
    {cites.length > 0 && <div className="crm-source">
      <span className="crm-source-head"><ShieldCheck size={13} aria-hidden="true" />
        Checked against {cites.length === 1 ? "the source" : `${cites.length} sources`}</span>
      <strong>{sourceTitle(cites[0].title)}{cites.length > 1 ? ` and ${cites.length - 1} more` : ""}</strong>
      {answer.quote && <q>{answer.quote}</q>}
    </div>}
    <Badge answer={answer} />
  </motion.section>;
}

function useSeconds(running: boolean) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  return seconds;
}

/** While the agent works: what it has done so far, one line each, and what it is doing now. */
function LiveActivity({ run }: { run: RunView }) {
  const seconds = useSeconds(true);
  const done = run.steps.flatMap((step) => step.calls).filter((call) => call.action.name !== "respond");
  return <div className="crm-activity" role="status">
    {done.map((call) => <p key={call.action.call_id} className="crm-activity-item">
      <Check size={13} aria-hidden="true" />{activityLine(call)}</p>)}
    <p className="crm-activity-now"><span className="thinking-pulse" aria-hidden="true" />
      {liveStatus(run)}<span className="crm-faint">{seconds} s</span></p>
  </div>;
}

function StatsLine({ stats }: { stats: StatsEvent }) {
  return <p className="crm-stats-line">
    {formatSeconds(stats.wall_ms)} · {formatInt(stats.prompt_tokens)} prompt tokens
    ({formatInt(stats.cached_tokens)} cached) · {formatInt(stats.completion_tokens)} generated
    · about {formatCost(stats.est_cost_usd)} of GPU time
  </p>;
}

/** Everything behind the answer, folded: the reasoning, each step with its inputs and results, the cost. */
function HowIGotThis({ run }: { run: RunView }) {
  const calls = run.steps.flatMap((step) => step.calls).filter((call) => call.action.name !== "respond");
  const time = run.stats ? ` · ${formatSeconds(run.stats.wall_ms)}` : "";
  const count = calls.length ? ` · ${calls.length} step${calls.length === 1 ? "" : "s"}` : "";
  return <Collapsible className="crm-howto">
    <CollapsibleTrigger className="crm-howto-trigger">
      How I got this{count}{time}
      <ChevronDown size={14} className="reasoning-chevron" />
    </CollapsibleTrigger>
    <CollapsibleContent>
      {run.answer?.reason && <p className="crm-howto-why"><span>Why</span>{run.answer.reason}</p>}
      <div className="crm-steps">
        {run.steps.filter((step) => step.calls.some((c) => c.action.name !== "respond") || step.thinking)
          .map((step, index, list) => <StepCard key={step.step} step={step} live={false}
            last={index === list.length - 1} />)}
      </div>
      {run.stats && <StatsLine stats={run.stats} />}
    </CollapsibleContent>
  </Collapsible>;
}

/** The case text under the question: three lines, with "Show all" only when the text is actually cut. */
function QuestionDetail({ text }: { text: string }) {
  const ref = useRef<HTMLParagraphElement>(null);
  const id = useId();
  const [open, setOpen] = useState(false);
  const [cut, setCut] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || open) return;
    const measure = () => setCut(el.scrollHeight > el.clientHeight + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [text, open]);
  return <>
    <p ref={ref} id={id} className={`crm-question-detail${open ? " open" : ""}`}>{text}</p>
    {(cut || open) && <button type="button" className="crm-detail-toggle" aria-expanded={open}
      aria-controls={id} onClick={() => setOpen(!open)}>{open ? "Show less" : "Show all"}</button>}
  </>;
}

/** One question and its answer, rendered live from the run's events. */
export function RunTimeline({ events, running, stopped = false, question, mode }: {
  events: CrmEvent[]; running: boolean; stopped?: boolean; question?: string; mode?: CrmMode;
}) {
  const run = buildRun(events);
  const start = run.start;
  const shownMode = start?.mode ?? mode;
  const customer = start?.context?.customer;
  const finished = run.answer !== null || run.done || run.errors.length > 0;
  return <article className="crm-run">
    <header className="crm-question">
      <h2>{start?.question ?? question ?? ""}</h2>
      {start?.detail && <QuestionDetail text={start.detail.replace(/\n\s*\n(\s*\n)+/g, "\n\n")} />}
      {shownMode === "customer" && <span className="crm-chip meta"><Lock size={12} aria-hidden="true" />
        Customer{customer ? `: ${customer}` : ""}, sees only their own records</span>}
    </header>
    {running && !finished && <LiveActivity run={run} />}
    {run.answer && <AnswerCard answer={run.answer} />}
    {run.errors.map((message, index) => <motion.div key={index} className="crm-error" role="alert"
      initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
      <TriangleAlert size={16} aria-hidden="true" />
      <div><strong>The run did not complete</strong><p>{message}</p></div>
    </motion.div>)}
    {stopped && <p className="crm-stopped"><CircleAlert size={13} aria-hidden="true" />Stopped before the run finished.</p>}
    {finished && (run.steps.length > 0 || run.answer?.reason) && <HowIGotThis run={run} />}
  </article>;
}
