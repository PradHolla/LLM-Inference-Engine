import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...values: ClassValue[]) {
  return twMerge(clsx(values));
}

export function formatMs(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "unavailable";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(1)} s`;
}

export function formatNumber(value: number | null | undefined, digits = 0) {
  return value == null || !Number.isFinite(value) ? "unavailable" : value.toFixed(digits);
}

export function initialTheme(): "cream" | "dark" {
  try {
    const saved = localStorage.getItem("llm-ui-theme");
    if (saved === "cream" || saved === "dark") return saved;
  } catch {}
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "cream";
}

export function messagePreference(chatId: number, key: "thinking" | "search", fallback: string | boolean) {
  try {
    const saved = localStorage.getItem(`llm-chat-${chatId}-${key}`);
    if (saved == null) return fallback;
    return key === "search" ? saved === "true" : saved;
  } catch {
    return fallback;
  }
}
