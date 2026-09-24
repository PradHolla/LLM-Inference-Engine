/** Turn inline `[n]` markers into citation nodes for sources 1..count; see code-notes.md.
 *  Usage: remarkPlugins={[...defaults, [remarkCitations, { count: 3 }]]} */
import type { Parent, Root, RootContent } from "mdast";

const MARKER = /\[(\d{1,3})\]/g;
const OPAQUE = new Set(["code", "inlineCode", "link", "linkReference", "definition", "html"]);

export type CitationOptions = { count: number };

function citation(n: number): RootContent {
  return { type: "citation", data: { hName: "sup", hProperties: {},
    hChildren: [{ type: "text", value: String(n) }] } } as unknown as RootContent;
}

export function splitCitations(value: string, count: number): RootContent[] | null {
  const parts: RootContent[] = [];
  let last = 0;
  for (const match of value.matchAll(MARKER)) {
    const n = Number(match[1]);
    if (n < 1 || n > count) continue;
    const start = match.index ?? 0;
    if (start > last) parts.push({ type: "text", value: value.slice(last, start) });
    parts.push(citation(n));
    last = start + match[0].length;
  }
  if (!parts.length) return null;
  if (last < value.length) parts.push({ type: "text", value: value.slice(last) });
  return parts;
}

function walk(node: Parent, count: number) {
  let changed = false;
  const next: RootContent[] = [];
  for (const child of node.children as RootContent[]) {
    if (child.type === "text") {
      const parts = splitCitations(child.value, count);
      if (parts) { next.push(...parts); changed = true; continue; }
    } else if ("children" in child && !OPAQUE.has(child.type)) {
      walk(child as Parent, count);
    }
    next.push(child);
  }
  if (changed) node.children = next as Parent["children"];
}

export function remarkCitations(options: CitationOptions) {
  return (tree: Root) => { if (options.count > 0) walk(tree, options.count); };
}
