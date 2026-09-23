import { forwardRef, memo, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import { MessageResponse } from "./ai-elements/message";
import { Reasoning, ReasoningContent, ReasoningTrigger } from "./ai-elements/reasoning";
import { SourceList } from "./ChatMessage";
import { formatMs } from "../lib/utils";
import type { MessageStats, Source, Theme } from "../types";

export type StreamingSnapshot = { content: string; thinking: string };
export type StreamingReplyHandle = {
  appendContent: (text: string) => void;
  appendThinking: (text: string) => void;
  snapshot: () => StreamingSnapshot;
};

export type PipelineState = {
  stage: "starting" | "searching" | "generating";
  query: string | null;
  sources: Source[] | null;
  stats: MessageStats | null;
  thinkingStreaming: boolean;
};

export const StreamingReply = memo(forwardRef<StreamingReplyHandle, {
  theme: Theme;
  pipeline: PipelineState;
  scrollRef: RefObject<HTMLDivElement | null>;
  pinnedRef: RefObject<boolean>;
}>(function StreamingReply({ theme, pipeline, scrollRef, pinnedRef }, ref) {
  const [content, setContent] = useState("");
  const [thinking, setThinking] = useState("");
  const contentRef = useRef("");
  const thinkingRef = useRef("");
  useImperativeHandle(ref, () => ({
    appendContent(text) { contentRef.current += text; setContent(contentRef.current); },
    appendThinking(text) { thinkingRef.current += text; setThinking(thinkingRef.current); },
    snapshot() { return { content: contentRef.current, thinking: thinkingRef.current }; },
  }), []);

  useEffect(() => {
    if (pinnedRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content, thinking, pipeline, pinnedRef, scrollRef]);

  const status = useMemo(() => {
    if (pipeline.stage === "searching") return pipeline.query
      ? `Searching the web for “${pipeline.query}”` : "Searching the web";
    if (pipeline.stage === "generating") return "Preparing a response";
    return "Starting request";
  }, [pipeline.query, pipeline.stage]);

  return <article className="message-row assistant-row live-row" aria-live="polite">
    <div className="assistant-message">
      <div className="pipeline-status"><span className="pipeline-dot" />{status}</div>
      {pipeline.sources && <SourceList sources={pipeline.sources} />}
      {thinking && <Reasoning isStreaming={pipeline.thinkingStreaming}
        defaultOpen={pipeline.thinkingStreaming}>
        <ReasoningTrigger />
        <ReasoningContent>{thinking}</ReasoningContent>
      </Reasoning>}
      {content ? <MessageResponse theme={theme} isAnimating>{content}</MessageResponse>
        : pipeline.stats && <div className="first-token-wait">First token {formatMs(pipeline.stats.ttft_ms)}</div>}
    </div>
  </article>;
}));

StreamingReply.displayName = "StreamingReply";
