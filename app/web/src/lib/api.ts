import type { AppConfig, Chat, ChatPayload } from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || detail;
    } catch {}
    throw new Error(String(detail));
  }
  return response.json() as Promise<T>;
}

export const getConfig = () => request<AppConfig>("/api/config");
export const getChats = () => request<Chat[]>("/api/chats");
export const getChat = (id: number) => request<ChatPayload>(`/api/chats/${id}`);
export const getHealth = () => request<{ gateway: boolean }>("/api/health");
export const createChat = () => request<Chat>("/api/chats", { method: "POST", body: "{}" });
export const renameChat = (id: number, title: string) => request<Chat>(`/api/chats/${id}`, {
  method: "PATCH", body: JSON.stringify({ title }),
});
export const deleteChat = (id: number) => request<{ ok: boolean }>(`/api/chats/${id}`, {
  method: "DELETE",
});
export const switchHead = (id: number, message_id: number) => request<ChatPayload>(
  `/api/chats/${id}/head`, { method: "POST", body: JSON.stringify({ message_id }) },
);
