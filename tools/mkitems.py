#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
mkitems.py -- generate the Phase 4 quality-eval item set. No GPU, no network, no dataset.
Three item families (math, gsm8k, longctx), each independently re-verified by --selftest.
See NOTES/code-notes.md for the design and the specific hazard it guards against.

  uv run tools/mkitems.py --out results/phase4-items.jsonl
  uv run tools/mkitems.py --out results/phase4-items.jsonl --selftest
"""
import argparse, json, random, re, sys

ANSWER_INT = ("End your reply with the final answer on its own line, "
              "in exactly this form:\nANSWER: <integer>")
ANSWER_STR = ("End your reply with the answer on its own line, "
              "in exactly this form:\nANSWER: <code>")

SUBJECTS = [
    ("depot", "crates"), ("orchard", "apples"), ("workshop", "bolts"),
    ("library", "volumes"), ("hatchery", "eggs"), ("quarry", "blocks"),
    ("bakery", "loaves"), ("nursery", "saplings"), ("mint", "coins"),
    ("aviary", "feathers"),
]
# Uniform "On day N" rather than weekday names: weekdays run out at k=10 and the dose
# curve needs to reach k=16. Still starts with "On " so the selftest parser is unchanged.
def day(i):
    return f"On day {i + 1}"
MULT_WORD = {2: "doubles", 3: "triples", 4: "quadruples"}
ORD_WORD = {3: "third", 4: "quarter", 5: "fifth", 6: "sixth"}

V_MIN, V_MAX = 3, 20000


def divisors(v, lo, hi):
    return [d for d in range(lo, hi + 1) if v % d == 0]


def gen_step(rng, v):
    """Return (sentence_fragment, new_value) or None if no op of this kind fits."""
    kind = rng.choice(["add", "sub", "mul", "div", "frac"])
    if kind == "add":
        n = rng.randint(5, 60)
        if v + n > V_MAX:
            return None
        return f"{n} more are added", v + n
    if kind == "sub":
        n = rng.randint(5, min(60, v - V_MIN)) if v - V_MIN >= 5 else None
        if n is None:
            return None
        return f"{n} are taken away", v - n
    if kind == "mul":
        m = rng.choice([2, 3, 4])
        if v * m > V_MAX:
            return None
        return f"the count {MULT_WORD[m]}", v * m
    if kind == "div":
        ds = [d for d in divisors(v, 2, 9) if v // d >= V_MIN]
        if not ds:
            return None
        d = rng.choice(ds)
        return f"they are shared equally into {d} groups and one group is kept", v // d
    if kind == "frac":
        ds = [d for d in divisors(v, 2, 6) if v * (d - 1) // d >= V_MIN]
        if not ds:
            return None
        d = rng.choice(ds)
        frag = ("half of them are taken away" if d == 2
                else f"one {ORD_WORD[d]} of them is taken away")
        return frag, v * (d - 1) // d
    return None


def make_math(rng, k, idx):
    place, unit = rng.choice(SUBJECTS)
    while True:
        v = rng.randint(12, 90)
        start = v
        lines, ok = [], True
        for i in range(k):
            for _ in range(40):
                step = gen_step(rng, v)
                if step:
                    break
            else:
                ok = False
                break
            frag, v = step
            lines.append(f"{day(i)}, {frag}.")
        if ok:
            break
    art = "An" if place[0] in "aeiou" else "A"
    body = (f"{art} {place} begins with {start} {unit}.\n"
            + "\n".join(lines)
            + f"\nHow many {unit} are there at the end?\n\n{ANSWER_INT}")
    return {"id": f"math-k{k}-{idx:04d}", "slice": "math", "k": k,
            "prompt": body, "answer": str(v), "start": start}


# --- longctx ------------------------------------------------------------------

VAULT_NAMES = ["Meridian", "Halcyon", "Ferrous", "Cobalt", "Perennial", "Thistle",
               "Lantern", "Orrery", "Bastion", "Kestrel"]
FILLER_T = [
    "The {a} committee reviewed the {b} inventory during the {c} quarter.",
    "Records from the {a} annex were transferred to the {b} archive without incident.",
    "A routine audit of the {a} ledger found the {b} totals consistent with the {c} report.",
    "Maintenance on the {a} corridor was deferred until the {b} inspection cycle.",
    "The {a} register lists {b} entries filed under the {c} classification.",
    "Staff rotation between the {a} and {b} wings continued on the usual schedule.",
]
FA = ["northern", "southern", "eastern", "western", "central", "lower", "upper", "outer"]
FB = ["seasonal", "provisional", "annual", "interim", "consolidated", "revised"]
FC = ["second", "third", "fourth", "preceding", "current", "subsequent"]


def make_longctx(rng, depth, idx, target_tokens=4000):
    names = rng.sample(VAULT_NAMES, 3)
    codes = []
    while len(codes) < 3:
        c = f"{rng.randint(0,9)}{rng.choice('QXKZVJ')}{rng.randint(0,9)}{rng.choice('QXKZVJ')}"
        if c not in codes:
            codes.append(c)
    facts = [f"The access code for the {n} vault is {c}." for n, c in zip(names, codes)]

    filler = [FILLER_T[rng.randrange(len(FILLER_T))].format(
        a=rng.choice(FA), b=rng.choice(FB), c=rng.choice(FC))
        for _ in range(600)]
    # bench.py's make_prompt uses 4 chars per token; keep the same convention so prompt
    # lengths are comparable with every other measurement in this project.
    doc, n = [], 0
    for s in filler:
        if n >= target_tokens * 4:
            break
        doc.append(s)
        n += len(s) + 1

    # target fact at `depth`, distractors at fixed offsets that never collide with it
    pos = max(1, min(len(doc) - 1, int(len(doc) * depth)))
    others = [p for p in (int(len(doc) * 0.05), int(len(doc) * 0.95))
              if abs(p - pos) > len(doc) * 0.1]
    while len(others) < 2:
        others.append(max(1, min(len(doc) - 1, (pos + len(doc) // 2) % len(doc))))
    for p, f in sorted(zip([pos] + others[:2], facts), key=lambda x: -x[0]):
        doc.insert(p, f)

    body = ("Read the following record and answer the question at the end.\n\n"
            + " ".join(doc)
            + f"\n\nQuestion: what is the access code for the {names[0]} vault?\n\n"
            + ANSWER_STR)
    return {"id": f"longctx-d{int(depth*100):02d}-{idx:04d}", "slice": "longctx",
            "depth": depth, "prompt": body, "answer": codes[0]}


def make_gsm8k(rows, n, rng):
    """Convert the published GSM8K test set into our item schema. Contamination is
    certain but does not matter here -- see NOTES/code-notes.md."""
    picked = rng.sample(rows, min(n, len(rows)))
    items = []
    for i, r in enumerate(picked):
        m = re.search(r"####\s*(.+?)\s*$", r["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "")     # 14 of 1319 finals are comma-grouped
        if not re.fullmatch(r"-?\d+", ans):
            continue
        items.append({"id": f"gsm8k-{i:04d}", "slice": "gsm8k",
                      "prompt": r["question"].strip() + "\n\n" + ANSWER_INT,
                      "answer": ans, "source_answer": r["answer"]})
    return items


def selftest_gsm8k(item):
    """Re-extract the final answer from the untouched source rationale."""
    m = re.search(r"####\s*(.+?)\s*$", item["source_answer"])
    if not m:
        return f"{item['id']}: no #### in source"
    if m.group(1).replace(",", "") != item["answer"]:
        return f"{item['id']}: source says {m.group(1)!r}, file says {item['answer']!r}"
    # No "answer leaks into the question" check -- tried and removed, see NOTES/code-notes.md.
    return None


# --- selftest: independently re-derive every maths answer from the TEXT ---------

PAT_START = re.compile(r"begins with (\d+) ")
PAT_STEPS = [
    (re.compile(r"(\d+) more are added"), lambda v, m: v + int(m[1])),
    (re.compile(r"(\d+) are taken away"), lambda v, m: v - int(m[1])),
    (re.compile(r"the count doubles"), lambda v, m: v * 2),
    (re.compile(r"the count triples"), lambda v, m: v * 3),
    (re.compile(r"the count quadruples"), lambda v, m: v * 4),
    (re.compile(r"shared equally into (\d+) groups"), lambda v, m: v // int(m[1])),
    (re.compile(r"half of them are taken away"), lambda v, m: v // 2),
    (re.compile(r"one third of them is taken away"), lambda v, m: v * 2 // 3),
    (re.compile(r"one quarter of them is taken away"), lambda v, m: v * 3 // 4),
    (re.compile(r"one fifth of them is taken away"), lambda v, m: v * 4 // 5),
    (re.compile(r"one sixth of them is taken away"), lambda v, m: v * 5 // 6),
]


def selftest_math(item):
    lines = item["prompt"].splitlines()
    v = int(PAT_START.search(lines[0]).group(1))
    steps = 0
    for ln in lines[1:]:
        if not ln.startswith(("On ", "The following ")):
            continue
        hit = 0
        for pat, fn in PAT_STEPS:
            m = pat.search(ln)
            if m:
                v, hit = fn(v, m), hit + 1
        if hit != 1:
            return f"{item['id']}: line matched {hit} operations, expected 1: {ln!r}"
        steps += 1
    if steps != item["k"]:
        return f"{item['id']}: parsed {steps} steps, k={item['k']}"
    if str(v) != item["answer"]:
        return f"{item['id']}: text replays to {v}, file says {item['answer']}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4-items.jsonl")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--k", default="2:30,4:60,8:120,16:150",
                    help="chain length:count pairs. Counts are DELIBERATELY UNEVEN -- at a "
                         "flat 90 per level, a true 2-point drop at k=16 shows up as ~1.8 "
                         "asymmetric items out of ~5 discordant, which no test can resolve. "
                         "k=2 is a ceiling with thinking on and needs almost "
                         "nothing; k=16 is where the effect lives and gets the items.")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--n-longctx", type=int, default=30, help="items per depth")
    ap.add_argument("--longctx-tokens", type=int, default=4000)
    ap.add_argument("--gsm8k", default="data/gsm8k-test.jsonl",
                    help="path to the published GSM8K test.jsonl; '' to skip")
    ap.add_argument("--n-gsm8k", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N sample items and exit")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = []
    for spec in args.k.split(","):
        k, n = (int(x) for x in spec.split(":"))
        items += [make_math(rng, k, i) for i in range(n)]
    for d in [float(x) for x in args.depths.split(",")]:
        items += [make_longctx(rng, d, i, args.longctx_tokens)
                  for i in range(args.n_longctx)]
    if args.gsm8k and args.n_gsm8k:
        try:
            rows = [json.loads(l) for l in open(args.gsm8k) if l.strip()]
        except FileNotFoundError:
            print(f"gsm8k source not found at {args.gsm8k}; fetch it with:\n"
                  "  curl -o data/gsm8k-test.jsonl https://raw.githubusercontent.com/"
                  "openai/grade-school-math/master/grade_school_math/data/test.jsonl",
                  file=sys.stderr)
            return 1
        items += make_gsm8k(rows, args.n_gsm8k, rng)

    if args.show:
        for it in items[:args.show]:
            print("=" * 70)
            print(it["id"], "->", it["answer"])
            print(it["prompt"][:1200])
        return 0

    bad = [e for e in (selftest_math(i) for i in items if i["slice"] == "math") if e]
    bad += [e for e in (selftest_gsm8k(i) for i in items if i["slice"] == "gsm8k") if e]
    if bad:
        print(f"SELFTEST FAILED: {len(bad)} items", file=sys.stderr)
        for e in bad[:10]:
            print("  " + e, file=sys.stderr)
        return 1
    n_math = sum(1 for i in items if i["slice"] == "math")
    n_g = sum(1 for i in items if i["slice"] == "gsm8k")
    print(f"selftest: {n_math} maths items replay from their own text to the stated answer")
    if n_g:
        print(f"selftest: {n_g} gsm8k answers re-extract from their untouched rationale")

    if args.selftest:
        return 0

    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items -> {args.out}")
    for k in sorted({i.get('k') for i in items if i['slice'] == 'math'}):
        ans = [int(i["answer"]) for i in items if i.get("k") == k]
        print(f"  math k={k}: {len(ans)} items, answers {min(ans)}-{max(ans)}")
    lc = [i for i in items if i["slice"] == "longctx"]
    print(f"  longctx: {len(lc)} items, prompt ~{len(lc[0]['prompt'])//4} tokens")
    g = [i for i in items if i["slice"] == "gsm8k"]
    if g:
        print(f"  gsm8k:   {len(g)} items from the published test set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
