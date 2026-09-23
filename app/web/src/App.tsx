import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowDown, Menu, MessageSquareText, Plus } from "lucide-react";
import { ChatMessage } from "./components/ChatMessage";
import { Composer } from "./components/Composer";
import { DeleteDialog, RenameDialog, Sidebar } from "./components/Sidebar";
import { StreamingReply, type PipelineState, type StreamingReplyHandle } from "./components/StreamingReply";
import { Button } from "./components/ui/button";
import { TooltipProvider } from "./components/ui/tooltip";
import { createChat, deleteChat, getChat, getChats, getConfig, getHealth, renameChat,
  switchHead } from "./lib/api";
import type { AppConfig, Chat, ChatPayload, Message, Source, Theme } from "./types";

type StreamState = { id: string; pipeline: PipelineState };

function stored(key: string, fallback: string): string {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}

export function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [chat, setChat] = useState<Chat | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);
  const [gatewayUp, setGatewayUp] = useState(true);
  const [theme, setTheme] = useState<Theme>(() => {
    const value = document.documentElement.dataset.theme;
    return value === "dark" ? "dark" : "cream";
  });
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<Chat | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Chat | null>(null);
  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState("brief");
  const [search, setSearch] = useState(true);
  const [stream, setStream] = useState<StreamState | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [atBottom, setAtBottom] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [editingMessage, setEditingMessage] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const replyRef = useRef<StreamingReplyHandle | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const stopRequested = useRef(false);
  const editingParentRef = useRef<number | null | undefined>(undefined);

  const refreshList = useCallback(async () => {
    const list = await getChats();
    setChats(list);
    return list;
  }, []);

  const refreshChat = useCallback(async (id: number): Promise<ChatPayload> => {
    const result = await getChat(id);
    setChat(result.chat);
    setMessages(result.messages);
    return result;
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([getConfig(), getChats(), getHealth()]).then(([nextConfig, list, health]) => {
      if (!active) return;
      setConfig(nextConfig);
      setChats(list);
      setGatewayUp(health.gateway);
      const queryId = Number(new URLSearchParams(location.search).get("chat"));
      const first = list.find((item) => item.id === queryId) ?? list[0];
      setSelectedId(first?.id ?? null);
      setLoading(false);
    }).catch((error: unknown) => {
      if (active) { setStreamError(error instanceof Error ? error.message : "Could not load the chat service."); setLoading(false); }
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (selectedId == null) { setChat(null); setMessages([]); return; }
    let active = true;
    setLoading(true);
    getChat(selectedId).then((payload) => {
      if (!active) return;
      setChat(payload.chat);
      setMessages(payload.messages);
      setLoading(false);
      history.replaceState(null, "", `/ui/?chat=${selectedId}`);
    }).catch((error: unknown) => {
      if (active) { setStreamError(error instanceof Error ? error.message : "Could not open this conversation."); setLoading(false); }
    });
    const prefThinking = stored(`llm-chat-${selectedId}-thinking`, "");
    const prefSearch = stored(`llm-chat-${selectedId}-search`, "");
    setThinking(prefThinking || chats.find((item) => item.id === selectedId)?.thinking_default || config?.default_thinking || "brief");
    setSearch(prefSearch ? prefSearch === "true" : config?.search_default ?? true);
    setDraft("");
    setEditingMessage(null);
    editingParentRef.current = undefined;
    setStreamError(null);
    pinnedRef.current = true;
    setAtBottom(true);
    return () => { active = false; };
  }, [selectedId]);

  useEffect(() => {
    if (selectedId == null) return;
    try {
      localStorage.setItem(`llm-chat-${selectedId}-thinking`, thinking);
      localStorage.setItem(`llm-chat-${selectedId}-search`, String(search));
    } catch {}
  }, [selectedId, thinking, search]);

  useEffect(() => {
    let active = true;
    const probe = async () => {
      try { const result = await getHealth(); if (active) setGatewayUp(result.gateway); }
      catch { if (active) setGatewayUp(false); }
    };
    const timer = window.setInterval(() => void probe(), 15000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const setPageTheme = (next: Theme) => {
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("llm-ui-theme", next); } catch {}
  };

  const scrollToBottom = useCallback((force = false) => {
    const element = scrollRef.current;
    if (!element || (!force && !pinnedRef.current)) return;
    element.scrollTop = element.scrollHeight;
    setAtBottom(true);
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, stream, scrollToBottom]);

  const chooseChat = (id: number) => {
    if (stream) return;
    setSelectedId(id);
    setMobileOpen(false);
  };

  const handleNewChat = async () => {
    if (stream) return;
    try {
      const next = await createChat();
      setChats((current) => [next, ...current]);
      setChat(next);
      setMessages([]);
      setSelectedId(next.id);
      setMobileOpen(false);
      setStreamError(null);
    } catch (error) { setStreamError(error instanceof Error ? error.message : "Could not create a conversation."); }
  };

  const handleRename = async (title: string) => {
    if (!renameTarget) return;
    const updated = await renameChat(renameTarget.id, title);
    setChats((current) => current.map((item) => item.id === updated.id ? updated : item));
    if (selectedId === updated.id) setChat(updated);
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    const targetId = deleteTarget.id;
    await deleteChat(targetId);
    const remaining = await refreshList();
    if (selectedId === targetId) setSelectedId(remaining[0]?.id ?? null);
  };

  const parseEvent = (line: string): Record<string, unknown> | null => {
    if (!line.startsWith("data:")) return null;
    try { return JSON.parse(line.slice(5).trim()) as Record<string, unknown>; }
    catch { return null; }
  };

  const readStream = async (response: Response, controller: AbortController,
    onEvent: (event: Record<string, unknown>) => void) => {
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try { const body = await response.json(); detail = body.detail || body.error || detail; } catch {}
      throw new Error(String(detail));
    }
    if (!response.body) throw new Error("The server returned an empty stream.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const event = parseEvent(line);
          if (event) onEvent(event);
        }
      }
      buffer += decoder.decode();
      const event = parseEvent(buffer.trim());
      if (event) onEvent(event);
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally {
      try { await reader.cancel(); } catch {}
    }
  };

  const reloadAfterStream = async (id: number) => {
    for (let attempt = 0; attempt < 5; attempt++) {
      try { await refreshChat(id); await refreshList(); return; }
      catch { await new Promise((resolve) => window.setTimeout(resolve, 140)); }
    }
  };

  const runStream = async (path: string, body: Record<string, unknown>, optimistic?: Message) => {
    if (selectedId == null || stream) return;
    const chatId = selectedId;
    const controller = new AbortController();
    abortRef.current = controller;
    stopRequested.current = false;
    setStreamError(null);
    if (optimistic) setMessages((current) => [...current, optimistic]);
    const id = `${chatId}-${Date.now()}`;
    setStream({ id, pipeline: { stage: "starting", query: null, sources: null, stats: null, thinkingStreaming: false } });
    pinnedRef.current = true;
    const onEvent = (event: Record<string, unknown>) => {
      if (event.type === "status") {
        const stage = event.stage === "searching" ? "searching" : "generating";
        setStream((current) => current ? { ...current, pipeline: {
          ...current.pipeline, stage, query: typeof event.query === "string" ? event.query : current.pipeline.query,
          thinkingStreaming: current.pipeline.thinkingStreaming,
        } } : current);
      } else if (event.type === "sources") {
        const sources = Array.isArray(event.sources) ? event.sources as Source[] : [];
        setStream((current) => current ? { ...current, pipeline: { ...current.pipeline, sources } } : current);
      } else if (event.type === "reasoning" && typeof event.text === "string") {
        setStream((current) => current ? { ...current, pipeline: {
          ...current.pipeline, stage: "generating", thinkingStreaming: true,
        } } : current);
        replyRef.current?.appendThinking(event.text);
      } else if (event.type === "content" && typeof event.text === "string") {
        setStream((current) => current ? { ...current, pipeline: {
          ...current.pipeline, stage: "generating", thinkingStreaming: false,
        } } : current);
        replyRef.current?.appendContent(event.text);
      } else if (event.type === "stats") {
        setStream((current) => current ? { ...current, pipeline: {
          ...current.pipeline, stats: event.stats as PipelineState["stats"],
        } } : current);
      } else if (event.type === "title" && typeof event.title === "string") {
        setChat((current) => current ? { ...current, title: event.title as string } : current);
        setChats((current) => current.map((item) => item.id === chatId ? { ...item, title: event.title as string } : item));
      } else if (event.type === "error") {
        const message = typeof event.message === "string" ? event.message : "The request did not complete.";
        setStreamError(message.includes("ConnectError") || message.includes("ConnectError")
          ? "The model gateway could not be reached. Check that it is running and try again."
          : message);
      }
    };

    try {
      const response = await fetch(path, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify(body), signal: controller.signal,
      });
      await readStream(response, controller, onEvent);
    } catch (error) {
      if (!controller.signal.aborted) {
        setStreamError(error instanceof Error ? error.message : "The request failed. Please try again.");
      }
    } finally {
      abortRef.current = null;
      await reloadAfterStream(chatId);
      setStream(null);
      if (stopRequested.current) setNotice("Response stopped. The partial answer was saved.");
      else setNotice(null);
      stopRequested.current = false;
      window.setTimeout(() => setNotice(null), 4000);
    }
  };

  const send = () => {
    if (!draft.trim() || selectedId == null || stream) return;
    const content = draft.trim();
    setDraft("");
    const parentId = editingParentRef.current !== undefined
      ? editingParentRef.current : chat?.head_message_id ?? null;
    const temporary: Message = {
      id: -Date.now(), parent_id: parentId, role: "user", content, thinking: null,
      created_at: Date.now() / 1000, tokens: null, sibling_ids: [], stopped: false,
      sources: null, stats: null,
    };
    void runStream(`/api/chats/${selectedId}/send`, {
      content, thinking, search, parent_id: parentId,
    }, temporary);
    editingParentRef.current = undefined;
    setEditingMessage(null);
  };

  const stop = () => {
    if (!abortRef.current) return;
    stopRequested.current = true;
    abortRef.current.abort();
  };

  const handleEdit = (message: Message) => {
    setDraft(message.content);
    editingParentRef.current = message.parent_id;
    setEditingMessage(message.id);
    const index = messages.findIndex((item) => item.id === message.id);
    if (index >= 0) setMessages(messages.slice(0, index));
    requestAnimationFrame(() => document.querySelector<HTMLTextAreaElement>(".composer textarea")?.focus());
  };

  const cancelEdit = () => {
    editingParentRef.current = undefined;
    setEditingMessage(null);
    setDraft("");
    if (selectedId != null) void refreshChat(selectedId);
  };

  const handleBranch = async (messageId: number) => {
    if (selectedId == null || stream) return;
    try {
      const result = await switchHead(selectedId, messageId);
      setChat(result.chat);
      setMessages(result.messages);
      pinnedRef.current = true;
    } catch (error) { setStreamError(error instanceof Error ? error.message : "Could not switch branch."); }
  };

  const handleRegenerate = (message: Message) => {
    if (selectedId == null) return;
    const index = messages.findIndex((item) => item.id === message.id);
    if (index >= 0) setMessages(messages.slice(0, index));
    void runStream(`/api/chats/${selectedId}/regenerate`, { message_id: message.id, thinking, search });
  };

  const titleRename = useMemo(() => chats.find((item) => item.id === renameTarget?.id) ?? renameTarget,
    [chats, renameTarget]);

  return <TooltipProvider>
    <div className="app-shell">
      <Sidebar chats={chats} currentId={selectedId} onSelect={chooseChat} onNew={() => void handleNewChat()}
        collapsed={collapsed} onToggleCollapsed={() => setCollapsed((value) => !value)}
        mobileOpen={mobileOpen} onMobileOpenChange={setMobileOpen} theme={theme}
        onThemeChange={() => setPageTheme(theme === "cream" ? "dark" : "cream")}
        onRename={setRenameTarget} onDelete={setDeleteTarget} />
      <main className="chat-main">
        <header className="topbar">
          <Button className="mobile-menu" variant="ghost" size="icon" aria-label="Open conversations"
            onClick={() => setMobileOpen(true)}><Menu size={18} /></Button>
          <div className="topbar-title"><span>{chat?.title ?? "Inference Chat"}</span>
            {chat && <small>#{chat.id}</small>}</div>
          <div className={`service-status ${gatewayUp ? "online" : "offline"}`}>
            <span />{gatewayUp ? "Gateway ready" : "Gateway unavailable"}
          </div>
        </header>

        {!gatewayUp && <div className="health-banner" role="status">
          The model gateway is unavailable. Your conversations are safe; new replies will not start until it reconnects.
        </div>}
        {streamError && <div className="error-banner" role="alert">
          <span>{streamError}</span><button type="button" onClick={() => setStreamError(null)} aria-label="Dismiss error">×</button>
        </div>}

        <div className="conversation-scroll" ref={scrollRef} onScroll={(event) => {
          const element = event.currentTarget;
          const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
          pinnedRef.current = nearBottom;
          setAtBottom(nearBottom);
        }}>
          <div className={`conversation ${messages.length === 0 && !loading ? "empty-conversation" : ""}`}>
            {loading ? <div className="loading-state"><span className="loading-mark" />Opening conversation</div>
              : messages.length === 0 ? <motion.section className="welcome" initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24 }}>
                <div className="welcome-symbol"><MessageSquareText size={20} /></div>
                <p className="welcome-kicker">Private, local inference</p>
                <h1>What would you like to explore?</h1>
                <div className="prompt-suggestions">
                  {["Explain a difficult idea with an example", "Compare two approaches to a problem", "Help me draft or revise something"].map((prompt) =>
                    <button key={prompt} type="button" onClick={() => setDraft(prompt)}>{prompt}<Plus size={15} /></button>)}
                </div>
              </motion.section>
              : <AnimatePresence initial={false}>
                {messages.map((message) => <motion.div key={message.id} layout="position"
                  initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.16 }}>
                  <ChatMessage message={message} theme={theme} onBranch={(id) => void handleBranch(id)}
                    onEdit={handleEdit} onRegenerate={handleRegenerate} />
                </motion.div>)}
              </AnimatePresence>}
            {stream && <StreamingReply key={stream.id} ref={replyRef} theme={theme}
              pipeline={stream.pipeline} scrollRef={scrollRef} pinnedRef={pinnedRef} />}
            {notice && <div className="stream-notice" role="status">{notice}</div>}
            <div className="scroll-anchor" />
          </div>
        </div>
        {!atBottom && <Button className="jump-bottom" size="icon" variant="quiet"
          aria-label="Jump to latest message" onClick={() => { pinnedRef.current = true; scrollToBottom(true); }}>
          <ArrowDown size={17} />
        </Button>}
        <div className="composer-dock">
          {editingMessage != null && <div className="edit-branch-notice">
            <span>Editing a message to create a branch</span>
            <button type="button" onClick={cancelEdit}>Cancel</button>
          </div>}
          <Composer value={draft} onChange={setDraft} onSend={send} onStop={stop} busy={Boolean(stream)}
            thinking={thinking} thinkingOptions={config?.thinking_levels ?? []} onThinkingChange={setThinking}
            search={search} onSearchChange={setSearch} />
        </div>
      </main>
      <RenameDialog chat={titleRename} onOpenChange={(open) => { if (!open) setRenameTarget(null); }} onSave={handleRename} />
      <DeleteDialog chat={deleteTarget} onOpenChange={(open) => { if (!open) setDeleteTarget(null); }} onDelete={handleDelete} />
    </div>
  </TooltipProvider>;
}

export default App;
