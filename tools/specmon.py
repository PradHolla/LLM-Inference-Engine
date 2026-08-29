#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
specmon.py -- measure speculative-decoding acceptance off vLLM's Prometheus endpoint.

Phase 5's central quantity is the acceptance rate `a`, and it arrives from a source this
project has never validated. CLAUDE.md section 2: assume any new measurement tool is
wrong until proven otherwise. Hence `selftest`, and hence section 9a of
NOTES/phase5-spec-design.md, which says no Phase 5 number is used until this tool
reproduces a constructed floor (a -> 0) and a constructed ceiling (a -> 1).

WHY THE PER-POSITION CURVE IS THE POINT
  A scalar a=0.6 is consistent with "every position accepts 60%" and with "position 1
  accepts 95%, position 3 accepts 5%". Those imply opposite choices of k and only the
  second is visible in the curve. Reporting the scalar alone would hide the one thing
  that decides the parameter.

METRIC NAMES ARE DISCOVERED, NOT HARDCODED
  CLAUDE.md section 1: never quote a remembered number, and the same applies to a
  remembered API. vLLM's spec-decode counter names differ across releases, so this tool
  pattern-matches whatever the server exposes and prints what it bound to. `discover`
  shows the raw list. If a binding is missing, the derived value is None -- never 0,
  because 0 is a legitimate measurement and None is "not measured" (incident 28).

DERIVATIONS
  a  = accepted_draft_tokens / drafted_tokens
  L  = (accepted + drafts) / drafts       mean tokens emitted per verify step; the "+
                                          drafts" is the bonus token the verify pass
                                          always produces, even on total rejection
  pos_i = accepted_at_position_i / drafts

  E_iid = (1 - a^(k+1)) / (1 - a)         what L WOULD be if acceptance were i.i.d.
  Comparing E_iid to the measured L tests that assumption rather than inheriting it;
  they diverge exactly to the degree acceptance is correlated across positions.

  uv run tools/specmon.py discover  --url http://localhost:8000
  uv run tools/specmon.py wrap      --url http://localhost:8000 --label s3-bf16-r4 \
                                    --out results/phase5-spec.jsonl -- \
        uv run tools/bench.py --url http://localhost:8000 ...
  uv run tools/specmon.py selftest
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, time
from pathlib import Path

# ---------------------------------------------------------------- prometheus parsing

def parse_prom(text: str) -> dict[str, float]:
    """Prometheus text exposition -> {name{labels}: value}.

    Keys keep their label set verbatim so positional series stay distinguishable.
    Deliberately tolerant: unparseable lines are skipped, not fatal, because a single
    malformed series must not cost the whole sample.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # split on the LAST space: label values may legally contain spaces
        sp = line.rfind(" ")
        if sp < 0:
            continue
        key, raw = line[:sp].strip(), line[sp + 1:].strip()
        try:
            val = float(raw)
        except ValueError:
            continue
        if math.isnan(val):
            continue
        out[key] = val
    return out


def _name(key: str) -> str:
    return key.split("{", 1)[0]


def _labels(key: str) -> dict[str, str]:
    if "{" not in key or not key.rstrip().endswith("}"):
        return {}
    body = key[key.index("{") + 1: key.rindex("}")]
    labels: dict[str, str] = {}
    for part in body.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        labels[k.strip()] = v.strip().strip('"')
    return labels


# ---------------------------------------------------------------- name binding

# Ordered most-specific first: `per_pos` must be tested before the bare `accepted`
# pattern, or the positional series would be bound as the scalar total.
#
# NOT_A_COUNT exists because a binding test against a plausible older naming scheme
# captured `spec_decode_draft_acceptance_rate` -- a RATIO -- into the `accepted` role,
# where it would have been summed with a token count. No exception, just a number a few
# tenths too large. That is exactly the failure class CLAUDE.md section 2 is about, so
# roles that mean "a count of things" reject any name that announces itself as a rate.
NOT_A_COUNT = ("rate", "ratio", "efficiency", "per_second", "seconds", "_time")

def _is_count(n: str) -> bool:
    return not any(w in n for w in NOT_A_COUNT)

BINDINGS = [
    ("per_pos",  lambda n: "spec_decode" in n and "accept" in n and _is_count(n)
                           and ("per_pos" in n or "position" in n)),
    ("accepted", lambda n: "spec_decode" in n and "accept" in n and _is_count(n)),
    ("drafted",  lambda n: "spec_decode" in n and "draft_token" in n and _is_count(n)),
    ("drafts",   lambda n: "spec_decode" in n and "draft" in n and "token" not in n
                           and _is_count(n)),
]


def bind(sample: dict[str, float]) -> dict[str, list[str]]:
    """Map each role to the metric keys that satisfy it. Order matters: once a key is
    claimed by a more specific role it is not offered to a looser one."""
    bound: dict[str, list[str]] = {r: [] for r, _ in BINDINGS}
    claimed: set[str] = set()
    for role, pred in BINDINGS:
        for key in sample:
            if key in claimed:
                continue
            n = _name(key)
            if n.endswith(("_created", "_sum", "_bucket")):
                continue
            if pred(n):
                bound[role].append(key)
                claimed.add(key)
    return bound


def _total(sample: dict[str, float], keys: list[str]) -> float | None:
    if not keys:
        return None
    return sum(sample[k] for k in keys)


# ---------------------------------------------------------------- derivation

def derive(before: dict[str, float], after: dict[str, float], k: int | None = None) -> dict:
    """Acceptance statistics over the interval [before, after]. Every derived value is
    None when its inputs are absent or zero -- never 0.0, which would be indistinguishable
    from a real measurement of total rejection."""
    b = bind(after)
    delta = {key: after[key] - before.get(key, 0.0) for key in after}

    drafts   = _total(delta, b["drafts"])
    drafted  = _total(delta, b["drafted"])
    accepted = _total(delta, b["accepted"])

    # positional series, keyed by whatever label carries the index
    pos: dict[int, float] = {}
    for key in b["per_pos"]:
        lab = _labels(key)
        idx = None
        for cand in ("position", "pos", "le", "index"):
            if cand in lab:
                try:
                    idx = int(float(lab[cand]))
                except ValueError:
                    idx = None
                break
        if idx is not None:
            pos[idx] = pos.get(idx, 0.0) + delta[key]

    # k can be inferred rather than supplied, when both counters are present
    k_eff = k
    if k_eff is None and drafts and drafted is not None and drafts > 0:
        ratio = drafted / drafts
        if abs(ratio - round(ratio)) < 0.02:
            k_eff = int(round(ratio))

    a = (accepted / drafted) if (accepted is not None and drafted) else None
    L = ((accepted + drafts) / drafts) if (accepted is not None and drafts) else None
    pos_rate = {i: (pos[i] / drafts) for i in sorted(pos)} if (pos and drafts) else None

    e_iid = None
    if a is not None and k_eff:
        e_iid = (k_eff + 1) if a >= 1.0 else (1 - a ** (k_eff + 1)) / (1 - a)

    return {
        "drafts": drafts, "drafted_tokens": drafted, "accepted_tokens": accepted,
        "k_inferred": k_eff,
        "acceptance_rate": a,
        "mean_emitted_per_step": L,
        "acceptance_by_position": pos_rate,
        "e_iid": e_iid,
        "iid_ratio": (L / e_iid) if (L and e_iid) else None,
        "bound": {r: v for r, v in b.items() if v},
        "unbound": [r for r, _ in BINDINGS if not b[r]],
    }


def report(d: dict, label: str = "") -> str:
    def f(x, n=4):
        return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
    out = [f"=== specmon {label} ==="]
    if d["unbound"]:
        out.append(f"  WARNING unbound roles: {', '.join(d['unbound'])} "
                   f"-- derived values depending on them are n/a, not zero")
    out.append(f"  verify steps (drafts)   {f(d['drafts'], 0)}")
    out.append(f"  draft tokens proposed   {f(d['drafted_tokens'], 0)}")
    out.append(f"  draft tokens accepted   {f(d['accepted_tokens'], 0)}")
    out.append(f"  k (inferred)            {f(d['k_inferred'])}")
    out.append(f"  acceptance rate a       {f(d['acceptance_rate'])}")
    out.append(f"  mean emitted / step L   {f(d['mean_emitted_per_step'])}")
    out.append(f"  E[L] if i.i.d.          {f(d['e_iid'])}")
    out.append(f"  measured / i.i.d.       {f(d['iid_ratio'])}"
               "   (>1 = acceptance is positively correlated across positions)")
    if d["acceptance_by_position"]:
        out.append("  acceptance by position:")
        for i, v in d["acceptance_by_position"].items():
            bar = "#" * int(round(v * 40))
            out.append(f"    pos {i}  {v:.4f}  {bar}")
    else:
        out.append("  acceptance by position: n/a (no positional series exposed)")
    n = d["drafts"]
    if n is not None and n < 400:
        out.append(f"  UNRELIABLE: {n:.0f} verify steps, design section 6 requires >= 400")
    return "\n".join(out)


# ---------------------------------------------------------------- I/O

def fetch(url: str, timeout: float = 10.0) -> dict[str, float]:
    import httpx
    r = httpx.get(url.rstrip("/") + "/metrics", timeout=timeout)
    r.raise_for_status()
    return parse_prom(r.text)


# ---------------------------------------------------------------- selftest

def _fixture(drafts: int, k: int, accepted_per_pos: list[int]) -> dict[str, float]:
    """Build a metrics sample with a known answer. accepted_per_pos[i] = how many of the
    `drafts` rounds had position i accepted."""
    s = {
        "vllm:spec_decode_num_drafts_total": float(drafts),
        "vllm:spec_decode_num_draft_tokens_total": float(drafts * k),
        "vllm:spec_decode_num_accepted_tokens_total": float(sum(accepted_per_pos)),
    }
    for i, c in enumerate(accepted_per_pos):
        s[f'vllm:spec_decode_num_accepted_tokens_per_pos{{position="{i}"}}'] = float(c)
    return s


def selftest() -> int:
    fails: list[str] = []

    def chk(name, got, want, tol=1e-9):
        ok = (got is None and want is None) or (
            got is not None and want is not None and abs(got - want) <= tol)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
        if not ok:
            fails.append(name)

    print("-- parser")
    txt = ('# HELP x help text with spaces\n'
           '# TYPE x counter\n'
           'x 1\n'
           'y{a="1",b="two words"} 2.5e3\n'
           'z NaN\n'
           'malformed_no_value\n')
    p = parse_prom(txt)
    chk("parse plain", p.get("x"), 1.0)
    chk("parse labels+exponent", p.get('y{a="1",b="two words"}'), 2500.0)
    chk("NaN dropped", p.get("z"), None)
    chk("malformed dropped", p.get("malformed_no_value"), None)
    lab = _labels('y{a="1",b="two words"}')
    chk("label parse", 1.0 if lab.get("b") == "two words" else 0.0, 1.0)

    print("-- FLOOR: draft never accepted (design 9a)")
    d = derive({}, _fixture(1000, 3, [0, 0, 0]))
    chk("floor a", d["acceptance_rate"], 0.0)
    chk("floor L", d["mean_emitted_per_step"], 1.0)
    chk("floor k", float(d["k_inferred"]), 3.0)

    print("-- CEILING: every draft accepted (design 9a)")
    d = derive({}, _fixture(1000, 3, [1000, 1000, 1000]))
    chk("ceiling a", d["acceptance_rate"], 1.0)
    chk("ceiling L", d["mean_emitted_per_step"], 4.0)
    chk("ceiling e_iid", d["e_iid"], 4.0)

    print("-- MID: hand-computable")
    # 1000 drafts, k=3; pos accepts 800/500/200 -> accepted 1500, drafted 3000
    d = derive({}, _fixture(1000, 3, [800, 500, 200]))
    chk("mid a", d["acceptance_rate"], 1500 / 3000)
    chk("mid L", d["mean_emitted_per_step"], 2500 / 1000)
    chk("mid pos0", d["acceptance_by_position"][0], 0.8)
    chk("mid pos2", d["acceptance_by_position"][2], 0.2)
    # i.i.d. at a=0.5, k=3 predicts 1.875; measured 2.5 -> positively correlated
    chk("mid e_iid", d["e_iid"], 1.875)

    print("-- DELTA: counters are cumulative, only the interval counts")
    before = _fixture(1000, 3, [800, 500, 200])
    after = _fixture(2000, 3, [800 + 0, 500 + 0, 200 + 0])
    after["vllm:spec_decode_num_accepted_tokens_total"] = 1500.0  # no new acceptances
    d = derive(before, after)
    chk("delta a", d["acceptance_rate"], 0.0)
    chk("delta drafts", d["drafts"], 1000.0)

    print("-- EDGES: absent and zero must not be confused (incident 28)")
    d = derive({}, {"vllm:spec_decode_num_drafts_total": 0.0})
    chk("no drafts -> a is None", d["acceptance_rate"], None)
    chk("no drafts -> L is None", d["mean_emitted_per_step"], None)
    d = derive({}, {"vllm:num_requests_running": 3.0})
    chk("no spec metrics -> a is None", d["acceptance_rate"], None)
    chk("unbound reported", 1.0 if "accepted" in d["unbound"] else 0.0, 1.0)

    print("-- BINDING: a RATE must never bind to a COUNT role")
    b = bind({"vllm:spec_decode_num_accepted_tokens_total": 1.0,
              "vllm:spec_decode_draft_acceptance_rate": 0.73,
              "vllm:spec_decode_num_draft_tokens_total": 3.0})
    chk("rate excluded from accepted", float(len(b["accepted"])), 1.0)
    chk("rate not bound anywhere",
        0.0 if any("acceptance_rate" in k for v in b.values() for k in v) else 1.0, 1.0)

    print("-- BINDING: per_pos must not be captured by the scalar pattern")
    b = bind(_fixture(10, 3, [1, 2, 3]))
    chk("accepted bound once", float(len(b["accepted"])), 1.0)
    chk("per_pos bound thrice", float(len(b["per_pos"])), 3.0)

    print("-- ARGV: `--` splits flags from the wrapped command (the REMAINDER bug)")
    left, cmd = split_argv(["discover", "--url", "http://h:8931"])
    chk("no `--`: flags stay left", float(len(cmd)), 0.0)
    chk("no `--`: url not stolen",
        1.0 if left == ["discover", "--url", "http://h:8931"] else 0.0, 1.0)
    left, cmd = split_argv(["wrap", "--url", "http://h:8931", "--", "bench", "--rate", "4"])
    chk("split: left keeps --url", 1.0 if "--url" in left else 0.0, 1.0)
    chk("split: cmd is the tail", 1.0 if cmd == ["bench", "--rate", "4"] else 0.0, 1.0)
    chk("split: cmd flags not parsed as ours", 1.0 if "--rate" not in left else 0.0, 1.0)

    print()
    if fails:
        print(f"SELFTEST FAILED: {len(fails)} -- {', '.join(fails)}")
        return 1
    print("SELFTEST PASSED -- floor, ceiling and edges all reproduce")
    return 0


# ---------------------------------------------------------------- main

def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split on the first bare `--`: flags to the left, wrapped command to the right.

    NOT argparse.REMAINDER. REMAINDER is greedy from the first positional onward, so
    `specmon.py discover --url http://host:8931` bound cmd=['--url','http://host:8931']
    and left `--url` at its DEFAULT. The tool then scraped localhost:8000, got
    ConnectionRefused, and the error pointed at the network rather than at argv. On the
    box the default IS the right URL, so this would have hidden entirely and only
    resurfaced as a wrap-mode misparse. Incident 28's lesson: test the boundary you
    just invented.
    """
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    argv_left, argv_cmd = split_argv(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["discover", "snapshot", "delta", "wrap", "watch", "selftest"])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="append the derived record here as JSONL")
    ap.add_argument("--before", help="delta mode: snapshot file")
    ap.add_argument("--after", help="delta mode: snapshot file")
    ap.add_argument("--file", help="snapshot mode: write here")
    ap.add_argument("--k", type=int, default=None, help="override inferred k")
    ap.add_argument("--interval", type=float, default=10.0, help="watch mode seconds")
    ap.add_argument("--cmd", nargs="*", default=[],
                    help=argparse.SUPPRESS)   # populated from argv, see split_argv

    a = ap.parse_args(argv_left)
    a.cmd = argv_cmd

    if a.mode == "selftest":
        return selftest()

    if a.mode == "discover":
        s = fetch(a.url)
        hits = {k: v for k, v in s.items() if "spec" in _name(k)}
        print(f"=== {len(s)} series on {a.url}/metrics, {len(hits)} matching 'spec' ===")
        for k in sorted(hits):
            print(f"  {k} = {hits[k]}")
        if not hits:
            print("  NONE. Either speculative decoding is off, or this build names the")
            print("  counters differently -- grep the full list before assuming the former:")
            for k in sorted(s)[:40]:
                print(f"    {k}")
        print()
        b = bind(s)
        print("bound roles:")
        for role, keys in b.items():
            print(f"  {role:9s} {keys if keys else 'UNBOUND'}")
        return 0

    if a.mode == "snapshot":
        s = fetch(a.url)
        dest = a.file or f"specmon-{int(time.time())}.json"
        Path(dest).write_text(json.dumps(s))
        print(f"wrote {len(s)} series to {dest}")
        return 0

    if a.mode == "delta":
        if not (a.before and a.after):
            ap.error("delta needs --before and --after")
        before = json.loads(Path(a.before).read_text())
        after = json.loads(Path(a.after).read_text())
        d = derive(before, after, a.k)
        print(report(d, a.label))
        rec = {"label": a.label, "ts": time.time(), **d}
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        return 0

    if a.mode == "watch":
        prev = fetch(a.url)
        try:
            while True:
                time.sleep(a.interval)
                cur = fetch(a.url)
                print(report(derive(prev, cur, a.k), a.label or "interval"))
                prev = cur
        except KeyboardInterrupt:
            return 0

    # wrap
    cmd = a.cmd
    if not cmd:
        ap.error("wrap needs: -- <command ...>")
    before = fetch(a.url)
    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0
    after = fetch(a.url)
    d = derive(before, after, a.k)
    print()
    print(report(d, a.label or " ".join(cmd[:3])))
    print(f"  wrapped command exited {rc} after {elapsed:.1f}s")
    if a.out:
        rec = {"label": a.label, "ts": time.time(), "elapsed_s": elapsed,
               "cmd": cmd, "rc": rc, **d}
        with open(a.out, "a") as f:      # flush per record: a sweep is long enough to
            f.write(json.dumps(rec) + "\n")   # lose everything to a Ctrl-C
            f.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
