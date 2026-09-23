export type Theme = "cream" | "dark";

export type ThinkingLevel = {
  id: string;
  label: string;
  description: string;
};

export type AppConfig = {
  thinking_levels: ThinkingLevel[];
  default_thinking: string;
  search_default: boolean;
};

export type Source = {
  title: string;
  url: string;
  site: string;
};

export type MessageStats = {
  ttft_ms: number | null;
  e2e_ms: number | null;
  engine_ttft_ms: number | null;
  search_ms: number | null;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  completion_tokens: number | null;
  decode_tok_s: number | null;
  thinking_level: string;
  searched: boolean;
};

export type Message = {
  id: number;
  parent_id: number | null;
  role: "user" | "assistant";
  content: string;
  thinking: string | null;
  created_at: number;
  tokens: number | null;
  sibling_ids: number[];
  stopped: boolean;
  sources: Source[] | null;
  stats: MessageStats | null;
};

export type Chat = {
  id: number;
  title: string;
  created_at: number;
  updated_at: number;
  thinking_default: string;
  head_message_id: number | null;
};

export type ChatPayload = { chat: Chat; messages: Message[] };

export type StreamEvent = {
  type: string;
  [key: string]: unknown;
};
