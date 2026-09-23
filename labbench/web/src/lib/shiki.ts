import { createHighlighterCore, type LanguageRegistration, type TokensResult } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import type { CodeHighlighterPlugin } from "streamdown";
import githubDark from "@shikijs/themes/github-dark";
import githubLight from "@shikijs/themes/github-light";

const languageImports: Record<string, () => Promise<{ default: LanguageRegistration[] }>> = {
  bash: () => import("@shikijs/langs/bash"),
  c: () => import("@shikijs/langs/c"),
  cpp: () => import("@shikijs/langs/cpp"),
  csharp: () => import("@shikijs/langs/csharp"),
  css: () => import("@shikijs/langs/css"),
  diff: () => import("@shikijs/langs/diff"),
  go: () => import("@shikijs/langs/go"),
  html: () => import("@shikijs/langs/html"),
  java: () => import("@shikijs/langs/java"),
  javascript: () => import("@shikijs/langs/javascript"),
  json: () => import("@shikijs/langs/json"),
  jsonc: () => import("@shikijs/langs/jsonc"),
  markdown: () => import("@shikijs/langs/markdown"),
  python: () => import("@shikijs/langs/python"),
  rust: () => import("@shikijs/langs/rust"),
  sql: () => import("@shikijs/langs/sql"),
  toml: () => import("@shikijs/langs/toml"),
  tsx: () => import("@shikijs/langs/tsx"),
  typescript: () => import("@shikijs/langs/typescript"),
  yaml: () => import("@shikijs/langs/yaml"),
};

const aliases: Record<string, string> = {
  bat: "bash", console: "bash", js: "javascript", jsx: "javascript", sh: "bash",
  shell: "bash", shellscript: "bash", ts: "typescript", yml: "yaml",
};
const loaded = new Set<string>();
const pending = new Map<string, Promise<void>>();
const highlighterPromise = createHighlighterCore({
  engine: createJavaScriptRegexEngine({ forgiving: true }),
  themes: [githubLight, githubDark],
});
let highlighterInstance: Awaited<typeof highlighterPromise> | null = null;

function languageName(input: string): string | null {
  const name = input.trim().toLowerCase();
  const canonical = aliases[name] ?? name;
  return canonical in languageImports ? canonical : null;
}

async function loadLanguage(name: string) {
  if (loaded.has(name)) return;
  let task = pending.get(name);
  if (!task) {
    task = (async () => {
      const registration = await languageImports[name]();
      const highlighter = await highlighterPromise;
      highlighterInstance = highlighter;
      await highlighter.loadLanguage(registration.default);
      loaded.add(name);
    })();
    pending.set(name, task);
  }
  await task;
}

function tokenize(highlighter: Awaited<typeof highlighterPromise>, source: string,
  language: string): TokensResult {
  return highlighter.codeToTokens(source, {
    lang: language,
    themes: { light: "github-light", dark: "github-dark" },
  });
}

export const codeHighlighter: CodeHighlighterPlugin = {
  name: "shiki",
  type: "code-highlighter",
  getSupportedLanguages: () => Object.keys(languageImports),
  getThemes: () => [githubLight, githubDark],
  supportsLanguage: (language) => languageName(language) !== null,
  highlight(options, callback) {
    const language = languageName(options.language);
    if (!language) return null;
    if (loaded.has(language)) {
      try { return highlighterInstance ? tokenize(highlighterInstance, options.code, language) : null; }
      catch { return null; }
    }
    if (callback) {
      void loadLanguage(language).then(async () => {
        try { callback(tokenize(await highlighterPromise, options.code, language)); }
        catch {}
      }).catch(() => {});
    }
    return null;
  },
};
