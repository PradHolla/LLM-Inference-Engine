// Offline checks for citation rendering and streamed-text reveal. Usage: npm run check
import { createServer } from "vite";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

let failures = 0;
function check(label, condition, detail = "") {
  console.log(`  ${condition ? "PASS" : "FAIL"}  ${label}${detail && !condition ? ` - ${detail}` : ""}`);
  if (!condition) failures++;
}

const server = await createServer({ configFile: new URL("../vite.config.ts", import.meta.url).pathname,
  server: { middlewareMode: true }, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: [] } });
try {
  const { MessageResponse } = await server.ssrLoadModule("/src/components/ai-elements/message.tsx");
  const { nextReveal } = await server.ssrLoadModule("/src/lib/reveal.ts");
  const sources = [
    { title: "First", url: "https://one.example/a", site: "one.example" },
    { title: "Second", url: "https://two.example/b", site: "two.example" },
  ];
  const render = (text, list) => renderToStaticMarkup(createElement(MessageResponse,
    { theme: "cream", sources: list }, text));
  const links = (html) => [...html.matchAll(/<sup class="citation"><a href="([^"]+)"[^>]*>(\d+)<\/a><\/sup>/g)]
    .map((m) => `${m[2]}=${m[1]}`);

  let html = render("Rates rose [1] and fell [2].", sources);
  check("in-range markers become links to the matching source",
    links(html).join(" ") === "1=https://one.example/a 2=https://two.example/b", html);
  check("citation links open in a new tab", /<sup class="citation"><a [^>]*target="_blank"/.test(html), html);
  check("surrounding text is kept", html.includes("Rates rose") && html.includes("and fell"), html);

  html = render("Only [3] and [0] here.", sources);
  check("out-of-range markers stay as text", links(html).length === 0 && html.includes("[3]") && html.includes("[0]"), html);

  html = render("See [1] here.", null);
  check("no sources leaves markers as text", links(html).length === 0 && html.includes("[1]"), html);
  html = render("See [1] here.", []);
  check("empty sources leaves markers as text", links(html).length === 0 && html.includes("[1]"), html);

  html = render("Inline `a[1]` and\n\n```python\nx = y[1]\n```\n\nthen [2].", sources);
  check("code spans and fenced code are untouched", links(html).join(" ") === "2=https://two.example/b" &&
    html.includes("a[1]") && html.includes("y[1]"), html);

  html = render("A [real link](https://z.example) and [1][2] together.", sources);
  check("adjacent markers both link, ordinary links untouched",
    links(html).join(" ") === "1=https://one.example/a 2=https://two.example/b" && html.includes("real link"), html);

  html = render("| a | b |\n|---|---|\n| x [2] | y |\n\n- item [1]\n", sources);
  check("markers inside tables and lists link", links(html).length === 2 && html.includes("<table"), html);

  const plain = render("Plain **bold** [1] text.", null);
  const before = renderToStaticMarkup(createElement(MessageResponse, { theme: "cream" }, "Plain **bold** [1] text."));
  check("a message without sources renders exactly as before", plain === before);

  const target = "Hello \u{1F600} world, ".repeat(40);
  let shown = 0, frames = 0, maxStep = 0, prefixOk = true, surrogateOk = true;
  while (shown < target.length && frames < 10000) {
    const next = nextReveal(target, shown, 16.7, false);
    maxStep = Math.max(maxStep, next - shown);
    const code = target.charCodeAt(next - 1);
    if (next < target.length && code >= 0xd800 && code <= 0xdbff) surrogateOk = false;
    if (next <= shown) prefixOk = false;
    shown = next; frames++;
  }
  check("paced reveal converges to the full text", shown === target.length, `${shown}/${target.length}`);
  check("paced reveal always advances", prefixOk);
  check("paced reveal never splits a surrogate pair", surrogateOk);
  check("a burst is spread over several frames", frames > 5 && maxStep < target.length / 3,
    `frames=${frames} maxStep=${maxStep}`);
  check("reduced motion reveals everything at once", nextReveal(target, 0, 16.7, true) === target.length);
  check("a long gap (hidden tab) catches up in one frame", nextReveal(target, 3, 5000, false) === target.length);
  check("nothing pending returns the full length", nextReveal("abc", 3, 16.7, false) === 3);
} finally {
  await server.close();
}
console.log(failures ? `  STREAM CHECKS FAILED (${failures})` : "  STREAM CHECKS PASSED");
process.exit(failures ? 1 : 0);
