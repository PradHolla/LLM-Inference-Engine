import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef,
  useState } from "react";
import type { RefObject } from "react";
import { flushSync } from "react-dom";
import { motion } from "motion/react";
import { MessageResponse, type CiteHandler } from "./ai-elements/message";
import { Reasoning, ReasoningContent, ReasoningTrigger } from "./ai-elements/reasoning";
import { SourceList } from "./ChatMessage";
import { nextReveal } from "../lib/reveal";
import { formatMs } from "../lib/utils";
import type { MessageStats, Plan, Source, Theme } from "../types";

export type StreamingSnapshot = { content: string; thinking: string };
export type StreamingReplyHandle = {
  appendContent: (text: string) => void;
  appendThinking: (text: string) => void;
  flush: () => void;
  snapshot: () => StreamingSnapshot;
};

export type PipelineState = {
  stage: "starting" | "planning" | "searching" | "generating";
  query: string | null;
  queries: string[] | null;
  plan: Plan | null;
  thinkingMode: string;
  sources: Source[] | null;
  stats: MessageStats | null;
  thinkingStreaming: boolean;
};

const FRAME_MS = 1000 / 60;

function prefersReducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export const StreamingReply = memo(forwardRef<StreamingReplyHandle, {
  theme: Theme;
  pipeline: PipelineState;
  scrollRef: RefObject<HTMLDivElement | null>;
  pinnedRef: RefObject<boolean>;
}>(function StreamingReply({ theme, pipeline, scrollRef, pinnedRef }, ref) {
  const [content, setContent] = useState("");
  const [thinking, setThinking] = useState("");
  const [cited, setCited] = useState<number | null>(null);
  const onCite = useCallback<CiteHandler>((index) => setCited(index), []);
  const contentRef = useRef("");
  const thinkingRef = useRef("");
  const shown = useRef({ content: 0, thinking: 0 });
  const frame = useRef<number | null>(null);
  const lastFrame = useRef<number | null>(null);
  const instant = useRef(prefersReducedMotion());

  const tick = useCallback((now: number) => {
    frame.current = null;
    const dt = lastFrame.current == null ? FRAME_MS : now - lastFrame.current;
    lastFrame.current = now;
    const state = shown.current;
    const nextThinking = nextReveal(thinkingRef.current, state.thinking, dt, instant.current);
    const nextContent = nextReveal(contentRef.current, state.content, dt, instant.current);
    if (nextThinking !== state.thinking || nextContent !== state.content) {
      state.thinking = nextThinking;
      state.content = nextContent;
      flushSync(() => {
        setThinking(thinkingRef.current.slice(0, nextThinking));
        setContent(contentRef.current.slice(0, nextContent));
      });
    }
    if (state.thinking < thinkingRef.current.length || state.content < contentRef.current.length) {
      frame.current = requestAnimationFrame(tick);
    } else {
      lastFrame.current = null;
    }
  }, []);

  const schedule = useCallback(() => {
    if (frame.current == null) frame.current = requestAnimationFrame(tick);
  }, [tick]);

  useImperativeHandle(ref, () => ({
    appendContent(text) { contentRef.current += text; schedule(); },
    appendThinking(text) { thinkingRef.current += text; schedule(); },
    flush() {
      if (frame.current != null) cancelAnimationFrame(frame.current);
      frame.current = null;
      lastFrame.current = null;
      shown.current = { content: contentRef.current.length, thinking: thinkingRef.current.length };
      setThinking(thinkingRef.current);
      setContent(contentRef.current);
    },
    snapshot() { return { content: contentRef.current, thinking: thinkingRef.current }; },
  }), [schedule]);

  useEffect(() => () => { if (frame.current != null) cancelAnimationFrame(frame.current); }, []);

  useLayoutEffect(() => {
    if (pinnedRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content, thinking, pipeline, pinnedRef, scrollRef]);

  const queries = pipeline.queries ?? pipeline.plan?.queries ?? [];
  const status = useMemo(() => {
    if (pipeline.stage === "planning") return "Deciding…";
    if (pipeline.stage === "searching") return queries.length || !pipeline.query
      ? "Searching the web" : `Searching the web for “${pipeline.query}”`;
    if (pipeline.stage === "generating") return "Preparing a response";
    return "Starting request";
  }, [pipeline.query, pipeline.stage, queries.length]);
  const thinkNote = pipeline.plan && pipeline.thinkingMode === "auto"
    ? pipeline.plan.think ? "Decided to think first" : "Decided to answer directly" : null;
  const showPlan = queries.length > 0 || thinkNote != null;

  return <article className="message-row assistant-row live-row" aria-live="polite">
    <div className="assistant-message">
      <div className="pipeline-status"><span className="pipeline-dot" />{status}</div>
      {showPlan && <motion.div className="plan-row" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18 }} aria-label="Plan">
        {queries.map((query, index) => <span className="query-chip" key={`${index}-${query}`}>{query}</span>)}
        {thinkNote && <span className="plan-note">{thinkNote}</span>}
      </motion.div>}
      {pipeline.sources && <SourceList sources={pipeline.sources} highlight={cited} />}
      {thinking && <Reasoning isStreaming={pipeline.thinkingStreaming}
        defaultOpen={pipeline.thinkingStreaming}>
        <ReasoningTrigger />
        <ReasoningContent>{thinking}</ReasoningContent>
      </Reasoning>}
      {content ? <MessageResponse theme={theme} isAnimating sources={pipeline.sources} onCite={onCite}>
        {content}</MessageResponse>
        : pipeline.stats && <div className="first-token-wait">First token {formatMs(pipeline.stats.ttft_ms)}</div>}
    </div>
  </article>;
}));

StreamingReply.displayName = "StreamingReply";
