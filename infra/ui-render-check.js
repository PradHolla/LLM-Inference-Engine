const fs = require("node:fs");
const path = require("node:path");

const repo = path.resolve(__dirname, "..");
let failures = 0;

function check(label, condition, detail = "") {
  console.log(`  ${String(condition ? "PASS" : "FAIL").padEnd(5)} ${label}${detail ? ` - ${detail}` : ""}`);
  if (!condition) failures++;
}

function sourceFiles(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? sourceFiles(target) : [target];
  });
}

function checkBuild(name, output) {
  const indexPath = path.join(repo, output, "index.html");
  const exists = fs.existsSync(indexPath);
  check(`${name} static index exists`, exists, output);
  if (!exists) return;
  const html = fs.readFileSync(indexPath, "utf8");
  const assetPaths = [...html.matchAll(/(?:src|href)="([^\"]+\.(?:js|css))"/g)].map((match) => match[1]);
  const localAssets = assetPaths.filter((asset) => asset.startsWith("/ui/assets/"));
  check(`${name} index references local built assets`, localAssets.length >= 2, `${localAssets.length} assets`);
  const allPresent = localAssets.every((asset) => {
    const target = path.join(repo, output, asset.replace(/^\/ui\//, ""));
    return fs.existsSync(target) && fs.statSync(target).size > 0;
  });
  check(`${name} referenced assets are non-empty`, allPresent);
  const cssPath = localAssets.find((asset) => asset.endsWith(".css"));
  const css = cssPath ? fs.readFileSync(path.join(repo, output, cssPath.replace(/^\/ui\//, "")), "utf8") : "";
  check(`${name} ships cream and dark style tokens`, css.includes("--page:") && css.includes("data-theme"));
  check(`${name} has no old vendored entrypoints`,
    !fs.existsSync(path.join(repo, output, "app.js")) && !fs.existsSync(path.join(repo, output, "vendor")));
}

checkBuild("chat", "app/ui");
checkBuild("labbench", "labbench/ui");

for (const name of ["app/web", "labbench/web"]) {
  const indexPath = path.join(repo, name, "index.html");
  const html = fs.existsSync(indexPath) ? fs.readFileSync(indexPath, "utf8") : "";
  check(`${name} chooses theme before paint`, html.includes("localStorage.getItem(\"llm-ui-theme\")") &&
    html.includes("prefers-color-scheme"));
  const files = sourceFiles(path.join(repo, name, "src"));
  const source = files.map((file) => fs.readFileSync(file, "utf8")).join("\n");
  check(`${name} keeps generated markup out of HTML strings`, !source.includes("dangerouslySetInnerHTML"));
  check(`${name} uses safe Markdown mode`, source.includes("skipHtml"));
  const emoji = /[\u{1F300}-\u{1FAFF}\u{1F000}-\u{1F2FF}\u{2600}-\u{27BF}\u{FE0F}]/u;
  check(`${name} source has no emoji`, !emoji.test(source));
}

const chatSource = sourceFiles(path.join(repo, "app/web/src")).map((file) => fs.readFileSync(file, "utf8")).join("\n");
check("chat settled messages keep memo boundaries", chatSource.includes("ChatMessage = memo") &&
  chatSource.includes("StreamingReply = memo"));
const benchSource = sourceFiles(path.join(repo, "labbench/web/src")).map((file) => fs.readFileSync(file, "utf8")).join("\n");
check("labbench live charts use uPlot", benchSource.includes("new uPlot") && benchSource.includes("uplot/dist/uPlot.min.css"));
check("labbench polling cadence is 2 Hz", benchSource.includes("setTimeout(poll, 500)"));

process.exitCode = failures ? 1 : 0;
