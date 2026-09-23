import { forwardRef, memo, useEffect, useImperativeHandle, useRef, useState, type RefObject } from "react";
import { Check, Copy } from "lucide-react";
import { Streamdown } from "streamdown";
import { codeHighlighter } from "../../lib/shiki";
import type { Theme, Turn } from "../../types";

export const MessageResponse = memo(function MessageResponse({ children, theme, isAnimating = false }: {
  children: string;
  theme: Theme;
  isAnimating?: boolean;
}) {
  const shiki = theme === "dark" ? "github-dark" : "github-light";
  return <Streamdown className="bench-markdown" plugins={{ code: codeHighlighter }} controls={{ code: { copy: true } }}
    isAnimating={isAnimating} skipHtml shikiTheme={[shiki, shiki]} components={{ img: () => null }}>
    {children}
  </Streamdown>;
}, (before, after) => before.children === after.children &&
  before.theme === after.theme && before.isAnimating === after.isAnimating);

export function ReasoningBlock({ text, live = false }: { text: string; live?: boolean }) {
  const [open, setOpen] = useState(live);
  const [seconds, setSeconds] = useState<number | null>(null);
  const startedAt = useRef<number | null>(live ? performance.now() : null);
  useEffect(() => {
    if (live) {
      startedAt.current ??= performance.now();
      setOpen(true);
    } else if (startedAt.current !== null) {
      setSeconds(Math.max(1, Math.ceil((performance.now() - startedAt.current) / 1000)));
      startedAt.current = null;
      setOpen(false);
    }
  }, [live]);
  return <details className="bench-reasoning" open={open} onToggle={(event) => {
    const target = event.currentTarget;
    setOpen(target.open);
    if (!target.open && live && startedAt.current !== null && seconds === null) {
      setSeconds(Math.max(1, Math.ceil((performance.now() - startedAt.current) / 1000)));
    }
  }}>
    <summary><span className={live ? "bench-thinking-dot active" : "bench-thinking-dot"} />
      {live ? "Thinking" : seconds == null ? "Thought" : `Thought for ${seconds}s`}</summary>
    <div className="bench-reasoning-copy">{text}</div>
  </details>;
}

export const BenchTurn = memo(function BenchTurn({ turn, theme }: { turn: Turn; theme: Theme }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(turn.content); setCopied(true); window.setTimeout(() => setCopied(false), 1200); }
    catch {}
  };
  return <article className={`bench-turn ${turn.role === "user" ? "bench-user" : "bench-assistant"}`}>
    <div className="turn-role">{turn.role === "user" ? "YOU" : "MODEL"}{turn.stopped && <span className="turn-stopped">STOPPED</span>}</div>
    <div className="turn-content">
      {turn.role === "user" ? <p className="bench-user-copy">{turn.content}</p> : <>
        {turn.reasoning && <ReasoningBlock text={turn.reasoning} />}
        <MessageResponse theme={theme}>{turn.content}</MessageResponse>
      </>}
      <button className="turn-copy" type="button" aria-label={copied ? "Copied message" : "Copy message"}
        title={copied ? "Copied" : "Copy message"} onClick={() => void copy()}>
        {copied ? <Check size={13} /> : <Copy size={13} />}
      </button>
    </div>
  </article>;
});

export type LiveTurnHandle = {
  appendContent: (text: string) => void;
  appendReasoning: (text: string) => void;
};

export const LiveTurn = forwardRef<LiveTurnHandle, {
  theme: Theme;
  scrollRef: RefObject<HTMLDivElement | null>;
  pinnedRef: RefObject<boolean>;
}>(function LiveTurn({ theme, scrollRef, pinnedRef }, ref) {
  const [content, setContent] = useState("");
  const [reasoning, setReasoning] = useState("");
  const contentRef = useRef("");
  const reasoningRef = useRef("");
  useImperativeHandle(ref, () => ({
    appendContent(text) { contentRef.current += text; setContent(contentRef.current); },
    appendReasoning(text) { reasoningRef.current += text; setReasoning(reasoningRef.current); },
  }), []);
  useEffect(() => {
    if (pinnedRef.current && scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [content, reasoning, pinnedRef, scrollRef]);
  return <article className="bench-turn bench-assistant bench-live-turn">
    <div className="turn-role"><span className="live-dot" /> LIVE</div>
    <div className="turn-content">
      {reasoning && <ReasoningBlock key="live-reasoning" text={reasoning} live={!content} />}
      {content ? <MessageResponse theme={theme} isAnimating>{content}</MessageResponse>
        : <div className="waiting-copy">Waiting for model output</div>}
    </div>
  </article>;
});
