import { memo, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowLeft, ArrowRight, Check, Copy, Pencil, RotateCw } from "lucide-react";
import { MessageResponse, type CiteHandler } from "./ai-elements/message";
import { Reasoning, ReasoningContent, ReasoningTrigger } from "./ai-elements/reasoning";
import { Button } from "./ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { formatMs, formatNumber } from "../lib/utils";
import type { Message, MessageStats, Theme } from "../types";

function IconAction({ label, children, onClick }: {
  label: string;
  children: ReactNode;
  onClick: () => void;
}) {
  return <Tooltip>
    <TooltipTrigger><Button className="message-action" size="iconSm" variant="ghost"
      aria-label={label} onClick={onClick}>{children}</Button></TooltipTrigger>
    <TooltipContent>{label}</TooltipContent>
  </Tooltip>;
}

function yesNo(value: boolean | null | undefined) {
  return value == null ? "unavailable" : value ? "yes" : "no";
}

function StatsDetails({ stats, tokens }: { stats: MessageStats; tokens: number | null }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      ref.current?.querySelector("summary")?.focus();
    };
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  const items: [string, string][] = [
    ["App first token", formatMs(stats.ttft_ms)],
    ["App end to end", formatMs(stats.e2e_ms)],
    ["Engine first token", formatMs(stats.engine_ttft_ms)],
    ["Planner", formatMs(stats.plan_ms)],
    ["Planner fallback", yesNo(stats.plan_fallback)],
    ["Search", formatMs(stats.search_ms)],
    ["Searched", yesNo(stats.searched)],
    ["Thinking setting", stats.thinking_level ?? "unavailable"],
    ["Reasoned", yesNo(stats.think)],
    ["Summary used", yesNo(stats.summary_used)],
    ["History tokens", formatNumber(stats.history_tokens)],
    ["Prompt tokens", formatNumber(stats.prompt_tokens)],
    ["Cached tokens", formatNumber(stats.cached_tokens)],
    ["Completion tokens", formatNumber(stats.completion_tokens ?? tokens)],
    ["Max answer tokens", formatNumber(stats.budget_max_tokens)],
    ["Decode", stats.decode_tok_s == null ? "unavailable" : `${formatNumber(stats.decode_tok_s, 1)} tok/s`],
  ];
  return <details className="stats-details" ref={ref} open={open}>
    <summary onClick={(event) => { event.preventDefault(); setOpen((value) => !value); }}>Details</summary>
    <dl>{items.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>
  </details>;
}

export const ChatMessage = memo(function ChatMessage({
  message, theme, onBranch, onEdit, onRegenerate,
}: {
  message: Message;
  theme: Theme;
  onBranch: (id: number) => void;
  onEdit: (message: Message) => void;
  onRegenerate: (message: Message) => void;
}) {
  const [copied, setCopied] = useState(false);
  const [cited, setCited] = useState<number | null>(null);
  const onCite = useCallback<CiteHandler>((index) => setCited(index), []);
  const assistant = message.role === "assistant";
  const siblingIndex = message.sibling_ids.indexOf(message.id);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {}
  };

  return <article className={`message-row ${assistant ? "assistant-row" : "user-row"}`}>
    <div className={assistant ? "assistant-message" : "user-message"}>
      {assistant ? <>
        {message.sources?.length ? <SourceList sources={message.sources} highlight={cited} /> : null}
        {message.thinking && <Reasoning defaultOpen={false}>
          <ReasoningTrigger />
          <ReasoningContent>{message.thinking}</ReasoningContent>
        </Reasoning>}
        <MessageResponse theme={theme} sources={message.sources} onCite={onCite}>{message.content}</MessageResponse>
        {message.stopped && <div className="stopped-note">Stopped early</div>}
        {message.stats && <div className="answer-footer">
          <div className="stats-line" aria-label="Reply statistics">
            <span>First token <b>{formatMs(message.stats.ttft_ms)}</b></span>
            <span>{message.stats.decode_tok_s == null ? "unavailable" : `${formatNumber(message.stats.decode_tok_s, 1)} tok/s`}</span>
            <span>{message.tokens == null ? "Tokens unavailable" : `${message.tokens} tokens`}</span>
            {message.stats.cached_tokens ? <span>{message.stats.cached_tokens} cached</span> : null}
          </div>
          <StatsDetails stats={message.stats} tokens={message.tokens} />
        </div>}
      </> : <div className="user-copy">{message.content}</div>}

      <div className="message-tools">
        {message.sibling_ids.length > 1 && <div className="branch-switcher" aria-label="Message branch">
          <Button size="iconSm" variant="ghost" aria-label="Previous branch"
            onClick={() => onBranch(message.sibling_ids[(siblingIndex - 1 + message.sibling_ids.length) % message.sibling_ids.length])}>
            <ArrowLeft size={15} />
          </Button>
          <span>{siblingIndex + 1} / {message.sibling_ids.length}</span>
          <Button size="iconSm" variant="ghost" aria-label="Next branch"
            onClick={() => onBranch(message.sibling_ids[(siblingIndex + 1) % message.sibling_ids.length])}>
            <ArrowRight size={15} />
          </Button>
        </div>}
        <IconAction label={copied ? "Copied" : "Copy message"} onClick={copy}>
          {copied ? <Check size={15} /> : <Copy size={15} />}
        </IconAction>
        {!assistant && <IconAction label="Edit and branch" onClick={() => onEdit(message)}>
          <Pencil size={15} />
        </IconAction>}
        {assistant && <IconAction label="Regenerate reply" onClick={() => onRegenerate(message)}>
          <RotateCw size={15} />
        </IconAction>}
      </div>
    </div>
  </article>;
});

export function SourceList({ sources, highlight = null }: { sources: Message["sources"]; highlight?: number | null }) {
  if (!sources) return null;
  return <section className="source-section" aria-label="Web sources">
    <div className="source-heading">Sources</div>
    {sources.length === 0 ? <span className="source-empty">No sources were returned.</span>
      : <div className="source-grid">{sources.map((source, index) => <a className={`source-card${highlight === index + 1 ? " cited" : ""}`}
        href={source.url} target="_blank" rel="noreferrer" key={`${source.url}-${index}`}>
        <span className="source-index">{index + 1}</span>
        <span className="source-text"><strong>{source.title || source.site}</strong>
          <small>{source.site}</small></span>
        <span className="source-open" aria-hidden="true">↗</span>
      </a>)}</div>}
  </section>;
}
