import { useEffect, useRef } from "react";
import { Globe2, Send, Square } from "lucide-react";
import type { ThinkingLevel } from "../types";
import { Button } from "./ui/button";

export function Composer({
  value, onChange, onSend, onStop, busy, thinking, thinkingOptions, onThinkingChange,
  search, onSearchChange,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  busy: boolean;
  thinking: string;
  thinkingOptions: ThinkingLevel[];
  onThinkingChange: (value: string) => void;
  search: boolean;
  onSearchChange: (value: boolean) => void;
}) {
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const selected = thinkingOptions.find((option) => option.id === thinking);

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
          <label className={`search-control ${search ? "selected" : ""}`}>
            <input type="checkbox" role="switch" checked={search} disabled={busy}
              onChange={(event) => onSearchChange(event.target.checked)} />
            <Globe2 size={15} />
            <span>Search</span>
            <span className="switch-track" aria-hidden="true"><span /></span>
          </label>
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
