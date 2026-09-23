export async function get<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function valueText(value: unknown, digits = 1, unit = ""): string {
  if (!isNumber(value)) return "unavailable";
  return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value)}${unit}`;
}

export function unavailable(reason?: string | null): string {
  return reason ? `unavailable - ${reason}` : "unavailable - no value returned";
}
