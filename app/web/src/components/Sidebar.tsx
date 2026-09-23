import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Check, ChevronLeft, ChevronRight, MessageSquarePlus, Moon, Pencil,
  Search, Sun, Trash2, X } from "lucide-react";
import type { Chat, Theme } from "../types";
import { Button } from "./ui/button";
import { Dialog, DialogContent } from "./ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

function IconButton({ label, children, onClick }: {
  label: string;
  children: ReactNode;
  onClick: () => void;
}) {
  return <Tooltip>
    <TooltipTrigger><Button variant="ghost" size="iconSm" aria-label={label} onClick={onClick}>
      {children}
    </Button></TooltipTrigger>
    <TooltipContent>{label}</TooltipContent>
  </Tooltip>;
}

function ChatList({ chats, currentId, onSelect, collapsed, onRename, onDelete }: {
  chats: Chat[];
  currentId: number | null;
  onSelect: (id: number) => void;
  collapsed: boolean;
  onRename: (chat: Chat) => void;
  onDelete: (chat: Chat) => void;
}) {
  const [filter, setFilter] = useState("");
  const filtered = useMemo(() => {
    const query = filter.trim().toLowerCase();
    return chats.filter((chat) => !query || `#${chat.id} ${chat.title}`.toLowerCase().includes(query));
  }, [chats, filter]);

  return <>
    {!collapsed && <label className="chat-filter">
      <Search size={15} aria-hidden="true" />
      <input aria-label="Search conversations" placeholder="Find a conversation" value={filter}
        onChange={(event) => setFilter(event.target.value)} />
      {filter && <button type="button" aria-label="Clear search" onClick={() => setFilter("")}><X size={14} /></button>}
    </label>}
    <nav className="chat-list" aria-label="Conversations">
      {filtered.map((chat) => <div key={chat.id} className={`chat-list-item ${chat.id === currentId ? "active" : ""}`}>
        <button className="chat-select" type="button" onClick={() => onSelect(chat.id)}
          aria-current={chat.id === currentId ? "page" : undefined} title={`#${chat.id} ${chat.title}`}>
          <span className="chat-number">#{chat.id}</span>
          {!collapsed && <span className="chat-name">{chat.title}</span>}
        </button>
        {!collapsed && <div className="chat-row-actions">
          <IconButton label="Rename conversation" onClick={() => onRename(chat)}><Pencil size={14} /></IconButton>
          <IconButton label="Delete conversation" onClick={() => onDelete(chat)}><Trash2 size={14} /></IconButton>
        </div>}
      </div>)}
      {!filtered.length && !collapsed && <p className="sidebar-empty">No matching conversations</p>}
    </nav>
  </>;
}

export function Sidebar({
  chats, currentId, onSelect, onNew, collapsed, onToggleCollapsed, mobileOpen,
  onMobileOpenChange, theme, onThemeChange, onRename, onDelete,
}: {
  chats: Chat[];
  currentId: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  mobileOpen: boolean;
  onMobileOpenChange: (open: boolean) => void;
  theme: Theme;
  onThemeChange: () => void;
  onRename: (chat: Chat) => void;
  onDelete: (chat: Chat) => void;
}) {
  const contents = (isMobile: boolean) => <div className={`sidebar-inner ${collapsed && !isMobile ? "is-collapsed" : ""}`}>
    <div className="sidebar-brand-row">
      <div className="brand-mark" aria-hidden="true">I</div>
      {(!collapsed || isMobile) && <div className="brand-copy"><strong>Inference</strong><span>Chat</span></div>}
      {!isMobile && <IconButton label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        onClick={onToggleCollapsed}>{collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}</IconButton>}
      {isMobile && <IconButton label="Close conversations" onClick={() => onMobileOpenChange(false)}>
        <X size={16} />
      </IconButton>}
    </div>
    <Button className="new-chat-button" onClick={onNew} variant="outline">
      <MessageSquarePlus size={16} />{(!collapsed || isMobile) && <span>New chat</span>}
    </Button>
    <ChatList chats={chats} currentId={currentId} onSelect={(id) => {
      onSelect(id);
      if (isMobile) onMobileOpenChange(false);
    }} collapsed={collapsed && !isMobile} onRename={onRename} onDelete={onDelete} />
    <div className="sidebar-bottom">
      <Button className="theme-button" variant="ghost" size={collapsed && !isMobile ? "icon" : "sm"}
        onClick={onThemeChange} aria-label={`Switch to ${theme === "dark" ? "cream" : "dark"} theme`}>
        {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        {(!collapsed || isMobile) && <span>{theme === "dark" ? "Cream theme" : "Dark theme"}</span>}
      </Button>
      {(!collapsed || isMobile) && <span className="sidebar-version">Local inference</span>}
    </div>
  </div>;

  return <>
    <aside className={`sidebar desktop-sidebar ${collapsed ? "collapsed" : ""}`}>
      {contents(false)}
    </aside>
    <Dialog open={mobileOpen} onOpenChange={onMobileOpenChange}>
      <DialogContent className="mobile-sidebar-dialog" aria-describedby="mobile-sidebar-description">
        <p id="mobile-sidebar-description" className="sr-only">Your saved conversations</p>
        {contents(true)}
      </DialogContent>
    </Dialog>
  </>;
}

export function RenameDialog({ chat, onOpenChange, onSave }: {
  chat: Chat | null;
  onOpenChange: (open: boolean) => void;
  onSave: (title: string) => Promise<void>;
}) {
  const [title, setTitle] = useState(chat?.title ?? "");
  const [saving, setSaving] = useState(false);
  useEffect(() => setTitle(chat?.title ?? ""), [chat]);
  const handleSave = async () => {
    if (!title.trim()) return;
    setSaving(true);
    try { await onSave(title.trim()); onOpenChange(false); }
    finally { setSaving(false); }
  };
  return <Dialog open={chat != null} onOpenChange={onOpenChange}>
    <DialogContent className="confirm-dialog">
      <h2>Rename conversation</h2>
      <input aria-label="Conversation title" autoFocus value={title}
        onChange={(event) => setTitle(event.target.value)} onKeyDown={(event) => {
          if (event.key === "Enter") void handleSave();
        }} />
      <div className="dialog-actions">
        <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="default" disabled={!title.trim() || saving} onClick={() => void handleSave()}>
          {saving ? "Saving" : <><Check size={15} /> Save</>}
        </Button>
      </div>
    </DialogContent>
  </Dialog>;
}

export function DeleteDialog({ chat, onOpenChange, onDelete }: {
  chat: Chat | null;
  onOpenChange: (open: boolean) => void;
  onDelete: () => Promise<void>;
}) {
  const [deleting, setDeleting] = useState(false);
  const handleDelete = async () => {
    setDeleting(true);
    try { await onDelete(); onOpenChange(false); }
    finally { setDeleting(false); }
  };
  return <Dialog open={chat != null} onOpenChange={onOpenChange}>
    <DialogContent className="confirm-dialog">
      <h2>Delete this conversation?</h2>
      <p>“{chat?.title}” and its messages will be removed.</p>
      <div className="dialog-actions">
        <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="danger" disabled={deleting} onClick={() => void handleDelete()}>
          <Trash2 size={15} />{deleting ? "Deleting" : "Delete"}
        </Button>
      </div>
    </DialogContent>
  </Dialog>;
}
