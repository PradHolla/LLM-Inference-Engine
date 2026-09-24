/** Per-frame reveal pacing for streamed text; see code-notes.md.
 *  Usage: shown = nextReveal(target, shown, dtMs, instant) inside a requestAnimationFrame tick. */

export const REVEAL_LAG_MS = 120;

export function nextReveal(target: string, shown: number, dtMs: number, instant: boolean): number {
  const backlog = target.length - shown;
  if (backlog <= 0) return target.length;
  if (instant) return target.length;
  const fraction = Math.min(1, Math.max(0, dtMs) / REVEAL_LAG_MS);
  let next = Math.min(target.length, shown + Math.max(1, Math.ceil(backlog * fraction)));
  // Never cut between the two halves of a surrogate pair.
  const code = target.charCodeAt(next - 1);
  if (next < target.length && code >= 0xd800 && code <= 0xdbff) next += 1;
  return next;
}
