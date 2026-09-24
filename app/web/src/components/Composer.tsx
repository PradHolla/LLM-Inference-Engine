import { useEffect, useRef } from "react";
import { Globe2, Send, Square } from "lucide-react";
import type { SearchMode, ThinkingLevel } from "../types";
import { Button } from "./ui/button";

export function Composer({
  value, onChange, onSend, onStop, busy, thinking, thinkingOptions, onThinkingChange,
  search, searchOptions, onSearchChange,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  busy: boolean;
  thinking: string;
  thinkingOptions: ThinkingLevel[];
  onThinkingChange: (value: string) => void;
  search: string;
  searchOptions: SearchMode[];
  onSearchChange: (value: string) => void;
}) {
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const selected = thinkingOptions.find((option) => option.id === thinking);
  const searchIndex = searchOptions.findIndex((option) => option.id === search);
  const moveSearch = (step: number) => {
    if (!searchOptions.length) return;
    const next = searchOptions[(Math.max(0, searchIndex) + step + searchOptions.length) % searchOptions.length];
    onSearchChange(next.id);
    requestAnimationFrame(() => document.querySelector<HTMLButtonElement>(
      `.search-segments [data-mode="${next.id}"]`)?.focus());
  };

  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, 220)}px`;
  }, [value]);

  return <div className="composer-wrap">
    <form className="composer" onSubmit={(event) => { event.preventDefault(); if (!busy) onSend(); }}>
      <textarea ref={areaRef} rows={1} value={value} disabled={busy}
        aria-label="Message" placeholder="Ask anything"
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            if (!busy) onSend();
          }
        }} />
      <div className="composer-controls">
        <div className="composer-options">
          <div className="search-control" role="radiogroup" aria-label="Web search"
            onKeyDown={(event) => {
              if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); moveSearch(1); }
              if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); moveSearch(-1); }
            }}>
            <Globe2 size={15} aria-hidden="true" />
            <span className="search-label">Search</span>
            <span className="search-segments">
              {searchOptions.map((option, index) => <button key={option.id} type="button" role="radio"
                data-mode={option.id} aria-checked={option.id === search} title={option.description}
                tabIndex={index === Math.max(0, searchIndex) ? 0 : -1} disabled={busy}
                onClick={() => onSearchChange(option.id)}>{option.label}</button>)}
            </span>
          </div>
          <label className="thinking-control">
            <span>Thinking</span>
            <select value={thinking} disabled={busy} aria-label="Thinking level"
              aria-describedby="thinking-description" onChange={(event) => onThinkingChange(event.target.value)}>
              {thinkingOptions.map((option) => <option key={option.id} value={option.id}
                title={option.description}>{option.label}</option>)}
            </select>
          </label>
          {selected && <span id="thinking-description" className="thinking-description">
            {selected.description}
          </span>}
        </div>
        {busy ? <Button className="send-button stop-button" variant="danger" onClick={onStop}
          aria-label="Stop generating"><Square size={15} fill="currentColor" /><span>Stop</span></Button>
          : <Button className="send-button" variant="default" type="submit" disabled={!value.trim()}>
            <Send size={15} /><span>Send</span>
          </Button>}
      </div>
    </form>
    <p className="composer-footnote">Enter to send · Shift+Enter for a new line</p>
  </div>;
}
