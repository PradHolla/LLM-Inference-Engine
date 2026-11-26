// Screenshots of the live CRM view for a design review, through the ssh tunnel on :8090.
// Usage: node scripts/shots.mjs <out-dir> [base-url]
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const out = process.argv[2];
const base = process.argv[3] ?? "http://localhost:8090/ui/?mode=crm";
if (!out) { console.error("usage: node scripts/shots.mjs <out-dir> [base-url]"); process.exit(2); }
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
async function page(width, height) {
  const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 });
  const p = await ctx.newPage();
  await p.goto(base, { waitUntil: "networkidle" });
  await p.waitForSelector(".crm-example, .crm-muted", { timeout: 20000 });
  return p;
}
const shot = async (p, name, full = false) => { await p.screenshot({ path: `${out}/${name}.png`, fullPage: full }); console.log("shot", name); };
// wait until the page holds n answers (or errors), then let the fade-in finish
const answers = async (p, n) => {
  await p.waitForFunction((k) => document.querySelectorAll(".crm-answer, .crm-error").length >= k, n, { timeout: 180000 });
  await p.waitForTimeout(600);
};
async function ask(p, question) {
  await p.fill("textarea[aria-label='Question']", question);
  await p.keyboard.press("Enter");
}

try {
  const p = await page(1440, 900);
  await shot(p, "01-welcome-staff");
  await p.click("button.crm-example >> nth=1");                 // routing: the tie case
  await p.waitForTimeout(1200);
  await shot(p, "02-routing-running");
  await answers(p, 1);
  await shot(p, "03-routing-answer");
  await p.click(".crm-howto-trigger");
  await p.waitForTimeout(400);
  await shot(p, "04-routing-howto-open", true);
  await p.click(".crm-howto-trigger");

  await ask(p, "What is the window for requesting a return after buying a product?");
  await p.waitForTimeout(4000);
  await shot(p, "05-knowledge-running");
  await answers(p, 2);
  await shot(p, "06-knowledge-answer");
  await ask(p, "Did the agent break policy on case 500Wt00000DE02HIAT?");
  await answers(p, 3);
  await shot(p, "07-policy-answer");

  const c = await page(1440, 900);
  await c.click("button[role='radio']:has-text('Customer')");
  await c.waitForTimeout(300);
  await shot(c, "08-welcome-customer");
  await c.click(".crm-suggestions button >> nth=0");
  await answers(c, 1);
  await shot(c, "09-customer-my-cases");
  await ask(c, "Can you check the order for Lucia Rossi?");
  await answers(c, 2);
  await shot(c, "10-customer-privacy");

  const m = await page(390, 844);
  await shot(m, "11-phone-welcome");
  await ask(m, "Who should handle a case?");
  await answers(m, 1);
  await shot(m, "12-phone-answer");
} finally {
  await browser.close();
}
console.log("SHOTS_DONE");
