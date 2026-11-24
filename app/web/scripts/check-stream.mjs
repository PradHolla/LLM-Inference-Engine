// Offline checks for citation rendering, streamed-text reveal and the CRM agent view. Usage: npm run check
import { createServer } from "vite";
import { readFileSync } from "node:fs";
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

  // CRM agent: timeline rendering, SSE reader and argument formatting.
  const crm = await server.ssrLoadModule("/src/lib/crm.ts");
  const { RunTimeline } = await server.ssrLoadModule("/src/components/crm/RunTimeline.tsx");
  const { TooltipProvider } = await server.ssrLoadModule("/src/components/ui/tooltip.tsx");
  const events = JSON.parse(readFileSync(new URL("./crm-fixture-handle_time_881.json", import.meta.url), "utf8"));
  const timeline = (list, props = {}) => renderToStaticMarkup(createElement(TooltipProvider, null,
    createElement(RunTimeline, { events: list, running: false, ...props })));
  const text = (html) => html.replace(/<!-- -->/g, "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ")
    .replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/&#x27;/g, "'").replace(/&quot;/g, "\"").replace(/&amp;/g, "&");

  let crmHtml = timeline(events);
  let crmText = text(crmHtml);
  check("timeline shows the action verbs", ["Summarize records", "Respond"].every((verb) => crmText.includes(verb)), crmText);
  check("filters render as field op value chips", crmText.includes("Field__c = Owner Assignment")
    && crmText.includes("CreatedDate >= 2022-09-01") && crmText.includes("ClosedDate is not null"), crmText);
  check("raw JSON is not shown until toggled", !crmHtml.includes("&quot;filters&quot;") && crmHtml.includes(">Raw<"));
  check("timeline shows the Matches the expected answer badge", crmText.includes("Matches the expected answer"), crmText);
  check("timeline shows the row counts", crmText.includes("118 rows") && crmText.includes("showing 2 of 118"), crmText);
  check("timeline shows the estimated cost with 4 decimals", crmText.includes("$0.0017") || crmText.includes("$0.0016"), crmText);
  check("timeline shows steps, time and tokens", crmText.includes("4.9 s") && crmText.includes("15,210 / 13,840 / 212"), crmText);
  check("timeline shows context chips", crmText.includes("Today 2022-01-18") && crmText.includes("Staff mode")
    && crmText.includes("Handle Time Policy"), crmText);
  check("Salesforce Ids use the monospace class", (crmHtml.match(/class="crm-id"/g) ?? []).length >= 4);
  check("one step card per step", (crmHtml.match(/class="crm-step"/g) ?? []).length === 3);
  check("a finished run shows no working indicator", !crmText.includes("Working"));

  const answerAt = events.findIndex((event) => event.type === "answer");
  const withAnswer = (patch) => events.map((event, index) => index === answerAt ? { ...event, ...patch } : event);
  crmText = text(timeline(withAnswer({ correct: false, expected: ["005AAA"] })));
  check("a wrong answer shows what was expected", crmText.includes("Does not match (expected: 005AAA)")
    && !crmText.includes("Matches the expected answer"), crmText);
  crmText = text(timeline(withAnswer({ correct: null, expected: null })));
  check("an ungraded answer shows no badge", !crmText.includes("Matches") && !crmText.includes("Does not match")
    && crmText.includes("005Wt000003NJ6gIAG"), crmText);

  const partial = events.slice(0, 3);
  crmText = text(timeline(partial, { running: true }));
  check("a running run shows the working indicator", crmText.includes("Working, step 1"), crmText);
  check("a pending action shows it is waiting", crmText.includes("Waiting for the result"), crmText);
  crmText = text(timeline([], { running: true, question: "Who owns case 5?", mode: "customer" }));
  check("before run_start the question and mode come from props",
    crmText.includes("Who owns case 5?") && crmText.includes("Customer mode"), crmText);

  const failing = [events[0], events[1], { ...events[2] }, { ...events[3], ok: false, rows_total: 0, rows_shown: 0,
    columns: [], preview: [], error: "unknown field 'Foo'" }, { type: "error", message: "gateway timeout" }];
  crmText = text(timeline(failing));
  check("a failed observation and an error event render", crmText.includes("Failed")
    && crmText.includes("unknown field 'Foo'") && crmText.includes("gateway timeout")
    && crmText.includes("The run did not complete"), crmText);
  crmText = text(timeline(partial, { stopped: true }));
  check("a stopped run says so", crmText.includes("Stopped before the run finished"), crmText);
  check("a stopped run does not spin on the unfinished action", crmText.includes("No result received")
    && !crmText.includes("Waiting for the result"), crmText);

  const thought = [events[0], events[1], { type: "thinking", step: 0, text: "Count owner changes first" }, ...events.slice(2)];
  crmHtml = timeline(thought);
  check("thinking is collapsed under its step", text(crmHtml).includes("Thinking") && !crmHtml.includes("Count owner changes first"));

  const wide = { ...events[3], columns: ["Id", "A", "B", "C", "D", "E"], rows_total: 1, rows_shown: 1,
    preview: [{ Id: "500Wt00000DDDfwIAH", A: 1, B: null, C: "x", D: "y", E: "z" }] };
  crmHtml = timeline([events[0], events[1], events[2], wide]);
  check("a single wide row renders as a field list", crmHtml.includes('class="crm-record"')
    && !crmHtml.includes('class="crm-table"') && text(crmHtml).includes("1 row"), crmHtml);

  const steps = crm.buildRun([events[0], { type: "step_start", step: 0 },
    { type: "action", step: 0, call_id: "a", name: "get_record", args: {} },
    { type: "action", step: 0, call_id: "b", name: "get_record", args: {} },
    { ...events[3], step: 0, call_id: "b" }]).steps;
  check("parallel calls pair observations by call_id", steps.length === 1 && steps[0].calls.length === 2
    && steps[0].calls[0].observation === null && steps[0].calls[1].observation?.call_id === "b");

  const argText = (name, args) => crm.formatArgs(name, args).map((item) => `${item.label}=${
    item.kind === "filters" ? item.filters.map(crm.filterText).join("|") : item.kind === "chips" ? item.chips.join("|") : item.text}`);
  check("summarize args fold the metric and having clause", argText("summarize_records", events[2].args).join(";")
    === "Object=CaseHistory__c;Filters=Field__c = Owner Assignment;Group by=CaseId__c;Metric=count;"
    + "Keep groups where=metric = 1;Sort=desc;Limit=200", argText("summarize_records", events[2].args).join(";"));
  check("avg_days_between reads as a sentence", argText("summarize_records", events[5].args)
    .includes("Metric=average days from CreatedDate to ClosedDate"));
  check("run_sql shows the query as code and respond shows nothing",
    crm.formatArgs("run_sql", { query: "SELECT 1" })[0].kind === "code" && crm.formatArgs("respond", { answer: "x" }).length === 0);
  check("query_records lists fields as chips", argText("query_records", { object: "Case", filters: [], fields: ["Id", "Status"],
    order_by: "", descending: true, limit: 5 }).join(";") === "Object=Case;Fields=Id|Status;Descending=yes;Limit=5");
  check("action names become readable verbs", crm.actionLabel("find_by_name") === "Find by name"
    && crm.actionLabel("search") === "Search text" && crm.actionLabel("do_thing") === "Do thing");
  check("Salesforce Id detection", crm.isSalesforceId("500Wt00000DDDfwIAH") && crm.isSalesforceId("005Wt000003NJ6g")
    && !crm.isSalesforceId("Owner Assignment") && !crm.isSalesforceId("abcdefghijklmno") && !crm.isSalesforceId(42));

  const sse = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")
    .replace('"Handle Time Policy"', '"Handle Time Policy é"');
  const noisy = `: keep-alive\r\nevent: ping\r\ndata: not json\r\ndata: {"type":"mystery"}\r\n${sse}`.replace(/\n\n/g, "\r\n\r\n");
  const bytes = new TextEncoder().encode(noisy);
  const chunks = [];
  for (let at = 0, size = 3; at < bytes.length; at += size, size = (size % 29) + 7) chunks.push(bytes.slice(at, at + size));
  const body = new ReadableStream({ start(controller) { chunks.forEach((chunk) => controller.enqueue(chunk)); controller.close(); } });
  const got = [];
  for await (const event of crm.readCrmEvents(body)) got.push(event);
  const expected = JSON.parse(sse.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5)).length
    ? JSON.stringify(events).replace('"Handle Time Policy"', '"Handle Time Policy é"') : "[]");
  check("the SSE reader yields every event in order across split chunks", chunks.length > 20
    && JSON.stringify(got) === JSON.stringify(expected), `${got.length}/${expected.length} events, ${chunks.length} chunks`);
  check("the SSE reader skips comments, junk and unknown types, and keeps multibyte text",
    got.every((event) => crm.CRM_EVENT_TYPES.includes(event.type)) && got[0].context.policy_title.endsWith("é"));
  const tail = new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode('data: {"type":"done"}')); controller.close(); } });
  const last = [];
  for await (const event of crm.readCrmEvents(tail)) last.push(event);
  check("the SSE reader flushes a final line with no newline", last.length === 1 && last[0].type === "done");
} finally {
  await server.close();
}
console.log(failures ? `  STREAM CHECKS FAILED (${failures})` : "  STREAM CHECKS PASSED");
process.exit(failures ? 1 : 0);
