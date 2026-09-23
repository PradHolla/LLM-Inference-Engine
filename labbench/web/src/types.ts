export type Theme = "cream" | "dark";

export type Turn = {
  id: number;
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  stopped?: boolean;
};

export type TraceEvent = { t_ms: number | null; bytes: number | null; chars: number | null };
export type Trace = {
  request_id?: string;
  status?: string;
  error?: string | null;
  ttft_ms?: number | null;
  e2e_ms?: number | null;
  n_events?: number | null;
  n_content_events?: number | null;
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
  completion_tokens?: number | null;
  tokens_per_event?: number | null;
  inter_event_p50_ms?: number | null;
  inter_event_p95_ms?: number | null;
  events?: TraceEvent[];
  upstream?: string;
};

export type BenchState = {
  backend?: { active?: string | null; status?: string; stage?: string | null;
    elapsed_s?: number | null; error?: string | null };
  config?: { model?: string | null; served_model?: string | null; served_model_error?: string | null;
    quantization?: string | null; kv_dtype?: string | null; max_model_len?: number | null;
    kv_tokens?: number | null; kv_gib?: number | null; spec?: string | null;
    launch_cmd?: string | null };
  engine?: { values?: Record<string, number | null>; bound?: Record<string, string[]>;
    n_series?: number | null; error?: string | null };
  gpu?: { devices?: Array<Record<string, number | null>>;
    processes?: Array<{ pid: number; used_mib: number }>;
    error?: string | null; processes_error?: string | null };
};

export type RenderedPrompt = { prompt?: string; n_tokens?: number | null; error?: string | null };
export type Prediction = { values?: Record<string, number | null>; text?: string; error?: string | null; note?: string };
export type Journal = { unit?: string | null; lines?: string[]; error?: string | null };
