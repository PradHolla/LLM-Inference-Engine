"use strict";

(function () {

const {
  useState, useEffect, useRef, useMemo, useCallback, memo, forwardRef, useImperativeHandle,
} = React;
const html = htm.bind(React.createElement);
const DEFAULT_THINKING = "brief";
const THEME_KEY = "llm-ui-theme";

function initialTheme() {
  try {
    const saved = window.localStorage.getItem(THEME_KEY);
    if (saved === "cream" || saved === "dark") return saved;
  } catch (_) { /* local storage can be unavailable in private contexts */ }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "cream";
}

function Icon({ name }) {
  const common = { viewBox: "0 0 24 24", width: "18", height: "18", fill: "none", stroke: "currentColor", strokeWidth: "1.8", strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": "true" };
  if (name === "plus") return html`<svg ...${common}><path d="M12 5v14M5 12h14" /></svg>`;
  if (name === "send") return html`<svg ...${common}><path d="m4 4 16 8-16 8 3-8-3-8Z" /><path d="M7 12h13" /></svg>`;
  if (name === "stop") return html`<svg ...${common}><rect x="7" y="7" width="10" height="10" rx="1" /></svg>`;
  if (name === "warning") return html`<svg ...${common}><path d="m12 4 8 15H4L12 4Z" /><path d="M12 9v4M12 16h.01" /></svg>`;
  if (name === "sun") return html`<svg ...${common}><circle cx="12" cy="12" r="3.5" /><path d="M12 2v2.5M12 19.5V22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M2 12h2.5M19.5 12H22M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" /></svg>`;
  return html`<svg ...${common}><path d="M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5Z" /></svg>`;
}

function applyInline(nodes, regex, keyBase, build) {
  const out = [];
  let counter = 0;
  for (const node of nodes) {
    if (typeof node !== "string") { out.push(node); continue; }
    regex.lastIndex = 0;
    let lastIndex = 0;
    let m;
    while ((m = regex.exec(node))) {
      if (m.index > lastIndex) out.push(node.slice(lastIndex, m.index));
      const built = build(m, keyBase + "-" + counter++);
      if (built.prefix) out.push(built.prefix);
      out.push(built.element);
      lastIndex = m.index + m[0].length;
      if (m[0].length === 0) regex.lastIndex++;
    }
    if (lastIndex < node.length) out.push(node.slice(lastIndex));
  }
  return out;
}

function inlineElements(text, keyBase) {
  let nodes = [text];
  nodes = applyInline(nodes, /`([^`]+)`/g, keyBase + "-code", (m, k) =>
    ({ element: html`<code key=${k}>${m[1]}</code>` }));
  nodes = applyInline(nodes, /\[([^\]]*)\]\(([^)\s]+)\)/g, keyBase + "-link", (m, k) =>
    ({ element: html`<a key=${k} href=${m[2]} target="_blank" rel="noopener">${m[1]}</a>` }));
  nodes = applyInline(nodes, /\*\*([^*]+)\*\*/g, keyBase + "-b", (m, k) =>
    ({ element: html`<strong key=${k}>${m[1]}</strong>` }));
  nodes = applyInline(nodes, /(^|[^*])\*([^*]+)\*(?!\*)/g, keyBase + "-i1", (m, k) =>
    ({ prefix: m[1], element: html`<em key=${k}>${m[2]}</em>` }));
  nodes = applyInline(nodes, /(^|[^_])_([^_]+)_(?!_)/g, keyBase + "-i2", (m, k) =>
    ({ prefix: m[1], element: html`<em key=${k}>${m[2]}</em>` }));
  return nodes;
}

function renderCodeBlockEl(segment, key) {
  const m = segment.match(/^([\w+-]*)\n([\s\S]*)$/);
  const body = m ? m[2] : segment;
  return html`<pre class="md-code" key=${key}><code>${body}</code></pre>`;
}

function renderTextBlockEls(segment, keyBase) {
  const lines = segment.split("\n");
  const blocks = [];
  let list = null;
  let para = [];
  let blockKey = 0;

  function flushPara() {
    if (!para.length) return;
    const children = [];
    para.forEach((lineNodes, idx) => {
      if (idx > 0) children.push(html`<br key=${keyBase + "-br" + blockKey + "-" + idx} />`);
      children.push(...lineNodes);
    });
    blocks.push(html`<p key=${keyBase + "-p" + blockKey++}>${children}</p>`);
    para = [];
  }
  function flushList() {
    if (!list) return;
    const Tag = list.tag;
    blocks.push(html`<${Tag} key=${keyBase + "-l" + blockKey++}>${list.items}</${Tag}>`);
    list = null;
  }

  lines.forEach((line, li) => {
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const ordered = line.match(/^\d+\.\s+(.*)$/);
    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (heading) {
      flushPara(); flushList();
      const Htag = "h" + heading[1].length;
      blocks.push(html`<${Htag} key=${keyBase + "-h" + blockKey++}>${inlineElements(heading[2], keyBase + "-h" + li)}</${Htag}>`);
    } else if (bullet) {
      flushPara();
      if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; }
      list.items.push(html`<li key=${keyBase + "-li" + li}>${inlineElements(bullet[1], keyBase + "-li" + li)}</li>`);
    } else if (ordered) {
      flushPara();
      if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; }
      list.items.push(html`<li key=${keyBase + "-li" + li}>${inlineElements(ordered[1], keyBase + "-li" + li)}</li>`);
    } else if (line.trim() === "") {
      flushPara(); flushList();
    } else {
      flushList();
      para.push(inlineElements(line, keyBase + "-ln" + li));
    }
  });
  flushPara(); flushList();
  return blocks;
}

function renderMarkdownElements(raw) {
  const parts = (raw || "").split("```");
  const blocks = [];
  parts.forEach((part, i) => {
    if (i % 2 === 1) blocks.push(renderCodeBlockEl(part, "code" + i));
    else blocks.push(...renderTextBlockEls(part, "blk" + i));
  });
  return blocks;
}

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response.json();
}

const MessageRow = memo(function MessageRow({ message }) {
  const blocks = useMemo(() => (
    message.role === "assistant" ? renderMarkdownElements(message.content) : null
  ), [message.content, message.role]);
  return html`
    <article class="message ${message.role}">
      <div class="message-role">${message.role}</div>
      ${message.role === "assistant" && message.thinking ? html`
        <details class="thinking-block">
          <summary>thinking (${message.thinking.length} chars)</summary>
          <div class="thinking-content">${message.thinking}</div>
        </details>` : null}
      <div class="message-body">${message.role === "assistant" ? blocks : message.content}</div>
    </article>`;
});

const MessageList = memo(function MessageList({ messages }) {
  return html`${messages.map((message) => html`<${MessageRow} key=${message.id} message=${message} />`)}`;
});

const StreamingMessage = memo(forwardRef(function StreamingMessage({ scrollRef, pinnedRef }, ref) {
  const [content, setContent] = useState("");
  const [reasoning, setReasoning] = useState("");
  const contentRef = useRef("");
  const reasoningRef = useRef("");

  useImperativeHandle(ref, () => ({
    appendContent(text) { contentRef.current += text; setContent(contentRef.current); },
    appendReasoning(text) { reasoningRef.current += text; setReasoning(reasoningRef.current); },
    snapshot() { return { content: contentRef.current, reasoning: reasoningRef.current }; },
  }), []);

  useEffect(() => {
    if (pinnedRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  });

  const blocks = useMemo(() => renderMarkdownElements(content), [content]);
  return html`
    <article class="message assistant">
      <div class="message-role">assistant</div>
      ${reasoning ? html`
        <details class="thinking-block">
          <summary>thinking (${reasoning.length} chars)</summary>
          <div class="thinking-content">${reasoning}</div>
        </details>` : null}
      <div class="message-body">${blocks}</div>
    </article>`;
}));

function App() {
  const [theme, setTheme] = useState(initialTheme);
  const [chats, setChats] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [chat, setChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [thinking, setThinking] = useState(DEFAULT_THINKING);
  const [search, setSearch] = useState(false);
  const [gateway, setGateway] = useState(null);
  const [newTitle, setNewTitle] = useState("");
  const [sending, setSending] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef(null);
  const streamRef = useRef(null);
  const scrollRef = useRef(null);
  const pinnedRef = useRef(true);
  const inputRef = useRef(null);
  const sequenceRef = useRef(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { window.localStorage.setItem(THEME_KEY, theme); } catch (_) { /* preference remains for this visit */ }
  }, [theme]);

  const loadChats = useCallback(async () => {
    const rows = await request("/api/chats");
    setChats(rows);
    setActiveId((current) => rows.some((row) => row.id === current) ? current : (rows[0] ? rows[0].id : null));
  }, []);

  const loadChat = useCallback(async (chatId) => {
    const data = await request("/api/chats/" + chatId);
    setChat(data.chat);
    setMessages(data.messages);
    setThinking(data.chat.thinking_default);
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      const data = await request("/api/health");
      setGateway(data.gateway);
    } catch (_) {
      setGateway(false);
    }
  }, []);

  useEffect(() => { loadChats().catch((err) => setError(String(err))); }, [loadChats]);
  useEffect(() => {
    if (activeId === null) { setChat(null); setMessages([]); return; }
    loadChat(activeId).catch((err) => setError(String(err)));
  }, [activeId, loadChat]);
  useEffect(() => {
    loadHealth();
    const timer = window.setInterval(loadHealth, 5000);
    return () => window.clearInterval(timer);
  }, [loadHealth]);
  useEffect(() => {
    if (pinnedRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const createChat = useCallback(async (event) => {
    event.preventDefault();
    const title = newTitle.trim();
    if (!title) return;
    try {
      const created = await request("/api/chats", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title, thinking_default: DEFAULT_THINKING }),
      });
      setNewTitle("");
      setActiveId(created.id);
      await loadChats();
    } catch (err) {
      setError(String(err));
    }
  }, [loadChats, newTitle]);

  const renameChat = useCallback(async (event) => {
    const title = event.target.value.trim();
    if (!chat || !title || title === chat.title) return;
    try {
      const updated = await request("/api/chats/" + chat.id, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title }),
      });
      setChat(updated);
      await loadChats();
    } catch (err) {
      setError(String(err));
    }
  }, [chat, loadChats]);

  const consumeStream = useCallback(async (response) => {
    if (!response.body) throw new Error("stream unavailable");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let done = false;
    while (!done) {
      const result = await reader.read();
      done = result.done;
      buffer += decoder.decode(result.value || new Uint8Array(), { stream: !done });
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).replace(/\r$/, "");
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data: ")) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch (_) { continue; }
        if (event.type === "content") streamRef.current?.appendContent(event.text || "");
        if (event.type === "reasoning") streamRef.current?.appendReasoning(event.text || "");
        if (event.type === "error") setError(event.message || "generation failed");
        if (event.type === "done") await loadChat(activeId);
      }
    }
  }, [activeId, loadChat]);

  const send = useCallback(async () => {
    const content = inputRef.current ? inputRef.current.value.trim() : "";
    if (!activeId || !content || sending) return;
    inputRef.current.value = "";
    setError("");
    setMessages((rows) => [...rows, { id: "local-user-" + ++sequenceRef.current, role: "user", content }]);
    setSending(true);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    try {
      const response = await fetch("/api/chats/" + activeId + "/send", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ content, thinking, search }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("HTTP " + response.status);
      await consumeStream(response);
      await loadChats();
    } catch (err) {
      if (err.name === "AbortError") {
        const partial = streamRef.current?.snapshot();
        if (partial && (partial.content || partial.reasoning)) {
          setMessages((rows) => [...rows, {
            id: "local-assistant-" + ++sequenceRef.current,
            role: "assistant",
            content: partial.content,
            thinking: partial.reasoning,
          }]);
        }
        window.setTimeout(() => { loadChat(activeId).catch(() => {}); loadChats().catch(() => {}); }, 250);
      } else {
        setError(String(err));
      }
    } finally {
      abortRef.current = null;
      setSending(false);
      setStreaming(false);
    }
  }, [activeId, consumeStream, loadChats, search, sending, thinking]);

  const stop = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
  }, []);

  const onScroll = useCallback(() => {
    const element = scrollRef.current;
    if (!element) return;
    pinnedRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 40;
  }, []);

  const onKeyDown = useCallback((event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(); }
  }, [send]);

  return html`
    <div class="app-shell">
      <aside class="sidebar">
        <div class="sidebar-heading">
          <span class="app-name">chat</span>
          <button class="icon-button theme-toggle" type="button" onClick=${() => setTheme((current) => current === "dark" ? "cream" : "dark")}
            aria-label=${theme === "dark" ? "Use cream theme" : "Use dark theme"} title=${theme === "dark" ? "Use cream theme" : "Use dark theme"}>
            <${Icon} name=${theme === "dark" ? "sun" : "moon"} />
          </button>
        </div>
        <form class="new-chat-form" onSubmit=${createChat}>
          <input aria-label="Chat title" value=${newTitle} onInput=${(event) => setNewTitle(event.target.value)} placeholder="New conversation" />
          <button class="new-chat-button" type="submit"><${Icon} name="plus" /> <span>New chat</span></button>
        </form>
        <nav class="conversation-list" aria-label="Conversations">
          ${chats.length ? chats.map((item) => html`
            <button class="conversation ${item.id === activeId ? "active" : ""}" key=${item.id}
              type="button" onClick=${() => setActiveId(item.id)}>${item.title}</button>`) :
            html`<div class="empty-list">No conversations</div>`}
        </nav>
      </aside>
      <main class="main">
        ${gateway === false ? html`<div class="gateway-banner" role="status"><${Icon} name="warning" />Gateway unavailable</div>` : null}
        <div class="chat-workspace">
          <div class="chat-toolbar">
            ${chat ? html`<input class="chat-title" aria-label="Chat title" defaultValue=${chat.title} onBlur=${renameChat} />` : null}
            <span class="toolbar-spacer"></span>
            <div class="thinking-control" role="group" aria-label="Thinking">
              ${[["off", "Off"], ["brief", "Brief"], ["full", "Full"]].map(([value, label]) => html`
                <button key=${value} type="button" class=${thinking === value ? "selected" : ""}
                  onClick=${() => setThinking(value)} disabled=${!chat || sending}>${label}</button>`)}
            </div>
            <label class="search-toggle"><input type="checkbox" checked=${search}
              onChange=${(event) => setSearch(event.target.checked)} disabled=${!chat || sending} /><span class="toggle-track"><span></span></span><span>Search</span></label>
          </div>
          <section class="message-scroll" ref=${scrollRef} onScroll=${onScroll}>
            <div class="message-column">
            ${chat ? (messages.length || streaming ? html`
              <${MessageList} messages=${messages} />
              ${streaming ? html`<${StreamingMessage} ref=${streamRef} scrollRef=${scrollRef} pinnedRef=${pinnedRef} />` : null}` :
              html`<p class="empty-chat">No messages</p>`) : html`<p class="empty-chat">Create a chat to begin</p>`}
            </div>
          </section>
          <section class="composer">
            ${error ? html`<div class="request-error" role="alert">${error}</div>` : null}
            <textarea ref=${inputRef} disabled=${!chat || sending} onKeyDown=${onKeyDown}
              placeholder="message"></textarea>
            <div class="composer-actions">
              ${sending ? html`<button class="stop icon-button" type="button" onClick=${stop} aria-label="Stop" title="Stop"><${Icon} name="stop" /></button>` : null}
              <button class="send icon-button" type="button" disabled=${!chat || sending} onClick=${send} aria-label="Send" title="Send"><${Icon} name="send" /></button>
            </div>
          </section>
        </div>
      </main>
    </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);

}());
