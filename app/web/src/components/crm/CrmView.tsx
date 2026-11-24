import { useCallback, useEffect, useRef, useState } from "react";
import { MotionConfig, motion } from "motion/react";
import { Briefcase, ChevronDown, Menu, Send, Square, Trash2 } from "lucide-react";
import { FALLBACK_AGENTS, askCrm, getCrmAgents, getCrmExamples, loadAgent, saveAgent, type AgentInfo, type AskRequest,
  type CrmAgent, type CrmEvent, type CrmExample, type CrmMode } from "../../lib/crm";
import { Button } from "../ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import { AgentPicker } from "./AgentPicker";
import { ExampleCards } from "./ExampleCards";
import { ModelPicker } from "./ModelPicker";
import { RunTimeline } from "./RunTimeline";

type RunStatus = "running" | "done" | "stopped" | "failed";
type RunRecord = { id: string; question: string; mode: CrmMode; agent: CrmAgent; events: CrmEvent[]; status: RunStatus };

const MODE_HINTS: Record<CrmMode, string> = {
  staff: "Answer as a support agent with access to all records",
  customer: "Answer as a customer asking about their own records",
};

const WELCOME: Record<CrmAgent, { title: string; sub: string }> = {
  service: {
    title: "Your service desk agent",
    sub: "It routes new cases, checks cases against company policy and answers from the knowledge base, one visible step at a time.",
  },
  general: {
    title: "Ask the agent about your CRM",
    sub: "Watch it plan, look up records and answer, one step at a time.",
  },
};

/** The CRM agent page: header controls, example questions, the run history and the question box. */
export function CrmView({ hidden, onOpenMenu }: { hidden: boolean; onOpenMenu: () => void }) {
  const [examples, setExamples] = useState<CrmExample[] | null>(null);
  const [examplesError, setExamplesError] = useState(false);
  const [agents, setAgents] = useState<AgentInfo[] | null>(null);
  const [agent, setAgent] = useState<CrmAgent>(loadAgent);
  const [mode, setMode] = useState<CrmMode>("staff");
  const [thinking, setThinking] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [question, setQuestion] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const pinnedRef = useRef(true);
  const counter = useRef(0);
  const active = runs.some((run) => run.status === "running");
  const blocked = active || switching;

  useEffect(() => {
    let live = true;
    getCrmAgents().then((list) => { if (live) setAgents(list); })
      .catch(() => { if (live) setAgents(FALLBACK_AGENTS); });
    return () => { live = false; };
  }, []);

  useEffect(() => {
    let live = true;
    setExamples(null);
    setExamplesError(false);
    getCrmExamples(agent).then((list) => { if (live) setExamples(list); })
      .catch(() => { if (live) setExamplesError(true); });
    return () => { live = false; };
  }, [agent]);

  const chooseAgent = (next: CrmAgent) => {
    if (active) return;
    setAgent(next);
    saveAgent(next);
  };

  useEffect(() => {
    const element = scrollRef.current;
    if (element && pinnedRef.current) element.scrollTop = element.scrollHeight;
  }, [runs]);

  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, 200)}px`;
  }, [question]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const patch = (id: string, change: (run: RunRecord) => RunRecord) =>
    setRuns((list) => list.map((run) => run.id === id ? change(run) : run));

  const startRun = async (body: AskRequest, label: string) => {
    if (abortRef.current) return;
    const id = `run-${++counter.current}`;
    const controller = new AbortController();
    abortRef.current = controller;
    pinnedRef.current = true;
    setRuns((list) => [...list, { id, question: label, mode: body.mode, agent: body.agent, events: [], status: "running" }]);
    let failed = false;
    try {
      for await (const event of askCrm(body, controller.signal)) {
        patch(id, (run) => ({ ...run, events: [...run.events, event] }));
        if (event.type === "error") failed = true;
      }
    } catch (reason) {
      if (!controller.signal.aborted) {
        failed = true;
        const message = reason instanceof Error ? reason.message : "The request failed.";
        patch(id, (run) => ({ ...run, events: [...run.events, { type: "error", message }] }));
      }
    } finally {
      abortRef.current = null;
      const stopped = controller.signal.aborted;
      patch(id, (run) => ({ ...run, status: stopped ? "stopped" : failed ? "failed" : "done" }));
    }
  };

  const ask = () => {
    const text = question.trim();
    if (!text || blocked) return;
    setQuestion("");
    void startRun({ question: text, example_id: null, mode, thinking, agent }, text);
  };

  const runExample = (example: CrmExample) => {
    if (blocked) return;
    setMode(example.mode);
    void startRun({ question: null, example_id: example.id, mode: example.mode, thinking, agent }, example.question);
  };

  const stop = () => abortRef.current?.abort();
  const clear = () => { if (!abortRef.current) setRuns([]); };
  const handleBusy = useCallback((busy: boolean) => setSwitching(busy), []);

  const empty = runs.length === 0;
  const pickers = examples && examples.length > 0
    ? <ExampleCards examples={examples} disabled={blocked} onRun={runExample} /> : null;

  return <MotionConfig reducedMotion="user">
    <main className="crm-main" hidden={hidden}>
      <header className="crm-header">
        <Button className="mobile-menu" variant="ghost" size="icon" aria-label="Open conversations"
          onClick={onOpenMenu}><Menu size={18} /></Button>
        <div className="crm-title">
          <span className="crm-title-icon"><Briefcase size={16} aria-hidden="true" /></span>
          <div><h1>CRM agent</h1><p>Answers questions from Salesforce service and sales data</p></div>
        </div>
        <div className="crm-controls">
          {agents && <AgentPicker agents={agents} value={agent} disabled={active} onChange={chooseAgent} />}
          <ModelPicker onBusyChange={handleBusy} />
          <div className="crm-segments" role="radiogroup" aria-label="Mode">
            {(["staff", "customer"] as CrmMode[]).map((value) => <button key={value} type="button" role="radio"
              aria-checked={mode === value} disabled={active} title={MODE_HINTS[value]}
              onClick={() => setMode(value)}>{value === "staff" ? "Staff" : "Customer"}</button>)}
          </div>
        </div>
      </header>

      {switching && <div className="health-banner" role="status">
        Switching model, about 1-2 minutes. Questions are paused until the new model is ready.
      </div>}

      <div className="conversation-scroll" ref={scrollRef} onScroll={(event) => {
        const element = event.currentTarget;
        pinnedRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
      }}>
        <div className={`crm-page ${empty ? "empty" : ""}`}>
          {empty && <motion.section className="crm-welcome" initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24 }}>
            <p className="welcome-kicker">Tool-using agent</p>
            <h2>{WELCOME[agent].title}</h2>
            <p className="crm-welcome-sub">{WELCOME[agent].sub}</p>
            {pickers}
            {!pickers && <p className="crm-muted">{examplesError ? "Example questions could not be loaded. You can still type a question below."
              : examples ? "No example questions yet. Type a question below." : "Loading examples"}</p>}
          </motion.section>}
          {runs.map((run) => <RunTimeline key={run.id} events={run.events} running={run.status === "running"}
            stopped={run.status === "stopped"} question={run.question} mode={run.mode}
            agent={run.agent} />)}
          {!empty && pickers && <Collapsible className="crm-more">
            <CollapsibleTrigger className="crm-more-trigger">Try another example
              <ChevronDown size={14} className="reasoning-chevron" /></CollapsibleTrigger>
            <CollapsibleContent>{pickers}</CollapsibleContent>
          </Collapsible>}
        </div>
      </div>

      <div className="composer-dock">
        <div className="composer-wrap">
          <form className="composer" onSubmit={(event) => { event.preventDefault(); ask(); }}>
            <textarea ref={areaRef} rows={1} value={question} disabled={blocked} aria-label="Question"
              placeholder={switching ? "Waiting for the model to finish switching" : "Ask a question about the CRM data"}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  ask();
                }
              }} />
            <div className="composer-controls">
              <div className="composer-options">
                <div className="search-control" role="radiogroup" aria-label="Show model thinking">
                  <span className="search-label">Thinking</span>
                  <span className="search-segments">
                    {[false, true].map((value) => <button key={String(value)} type="button" role="radio"
                      aria-checked={thinking === value} disabled={active}
                      onClick={() => setThinking(value)}>{value ? "On" : "Off"}</button>)}
                  </span>
                </div>
                {!empty && <Button className="crm-clear" variant="ghost" size="sm" onClick={clear} disabled={active}
                  aria-label="Clear runs"><Trash2 size={14} />Clear</Button>}
              </div>
              {active
                ? <Button className="send-button stop-button" variant="danger" onClick={stop}
                  aria-label="Stop the run"><Square size={15} fill="currentColor" /><span>Stop</span></Button>
                : <Button className="send-button" variant="default" type="submit"
                  disabled={!question.trim() || switching}><Send size={15} /><span>Run</span></Button>}
            </div>
          </form>
        </div>
      </div>
    </main>
  </MotionConfig>;
}
