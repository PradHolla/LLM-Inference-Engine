"""
probes.py -- read-only views of the box: engine metrics, GPU, units, logs.
No probe raises; one that cannot read its source returns an `error` field, because
a panel that goes blank is indistinguishable from a panel reporting zero.

  uv run --with httpx python labbench/probes.py     # selftest against fixtures
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import specmon  # noqa: E402  -- reused for Prometheus parsing, not re-implemented

GPU_FIELDS = ("memory.total", "memory.used", "utilization.gpu",
              "power.draw", "power.limit", "temperature.gpu")

# Series that are not the value they appear to be. `_created` is a unix timestamp of
# counter creation, `_bucket`/`_sum` are histogram internals, `external_` is the KV
# connector's cache rather than the local prefix cache, and `_by_reason` is a breakdown
# of a total that is bound separately -- summing both double-counts.
def _usable(n: str) -> bool:
    return not (n.endswith("_created") or n.endswith("_bucket") or n.endswith("_sum")
                or "external_" in n or "_by_reason" in n)


# Role -> predicate over a metric name. Discovered, never hardcoded: vLLM renames these
# between versions and a wrong binding reports a plausible number rather than an error.
BINDINGS = [
    ("running",        lambda n: _usable(n) and "num_requests_running" in n),
    ("waiting",        lambda n: _usable(n) and "num_requests_waiting" in n),
    ("kv_usage",       lambda n: _usable(n) and "cache_usage_perc" in n and "prefix" not in n),
    ("prefix_queries", lambda n: _usable(n) and "prefix_cache_queries" in n),
    ("prefix_hits",    lambda n: _usable(n) and "prefix_cache_hits" in n),
]

KV_PATTERNS = {
    "kv_tokens": r"GPU KV cache size:\s*([\d,]+)\s*tokens",
    "kv_gib": r"Available KV cache memory:\s*([\d.]+)\s*GiB",
    "max_concurrency": r"Maximum concurrency for\s*([\d,]+)\s*tokens",
}


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """Run a command, never raise. Returns (rc, combined output)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 127, f"{type(e).__name__}: {e}"


def parse_gpu(dev_csv: str, proc_csv: str) -> dict:
    """Parse the two nvidia-smi CSV queries into one dict."""
    out: dict = {"devices": [], "processes": []}
    for line in dev_csv.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(GPU_FIELDS):
            continue
        d = {}
        for k, v in zip(GPU_FIELDS, vals):
            try:
                d[k.replace(".", "_")] = float(v)
            except ValueError:
                d[k.replace(".", "_")] = None
        out["devices"].append(d)
    for line in proc_csv.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != 2:
            continue
        try:
            out["processes"].append({"pid": int(vals[0]), "used_mib": float(vals[1])})
        except ValueError:
            continue
    return out


def gpu() -> dict:
    """nvidia-smi device totals plus per-process memory."""
    rc1, dev = _run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
                     "--format=csv,noheader,nounits"])
    rc2, proc = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits"])
    if rc1 != 0:
        return {"error": dev.strip(), "devices": [], "processes": []}
    d = parse_gpu(dev, proc if rc2 == 0 else "")
    if rc2 != 0:
        d["processes_error"] = proc.strip()
    return d


def unit(name: str) -> dict:
    """systemctl state for one unit. `active` is what the backend switcher branches on."""
    rc, act = _run(["systemctl", "is-active", name])
    props = "MainPID,SubState,ActiveState,ExecMainStartTimestamp"
    rc2, show = _run(["systemctl", "show", name, f"--property={props}"])
    d: dict = {"unit": name, "active": act.strip() or "unknown"}
    if rc2 == 0:
        for line in show.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                d[k] = v
    else:
        d["error"] = show.strip()
    return d


def journal(name: str, lines: int = 200) -> dict:
    """Recent journal for a unit, newest last."""
    rc, out = _run(["journalctl", "-u", name, "-n", str(lines), "--no-pager", "-o", "cat"],
                   timeout=10.0)
    if rc != 0:
        return {"unit": name, "lines": [], "error": out.strip()}
    return {"unit": name, "lines": out.splitlines()}


def kv_from_log(text: str) -> dict:
    """Resolved KV budget from a vLLM startup log. Patterns match infra/vllm-launch.sh."""
    out: dict = {}
    for key, pat in KV_PATTERNS.items():
        m = re.findall(pat, text)
        if not m:
            continue
        raw = m[-1].replace(",", "")
        out[key] = float(raw) if "." in raw else int(raw)
    return out


def served_model(url: str) -> dict:
    """The id the engine will accept in a request. vLLM validates `model` and 404s on a
    wrong one; baseline and engine ignore it. Discovered, because the backend changes."""
    try:
        import httpx
        r = httpx.get(url.rstrip("/") + "/v1/models", timeout=5.0)
        if r.status_code != 200:
            return {"id": None, "error": f"/v1/models returned {r.status_code}"}
        data = (r.json() or {}).get("data") or []
        return {"id": (data[0].get("id") if data else None), "error": None}
    except Exception as e:
        return {"id": None, "error": f"{type(e).__name__}: {e}"}


def bind_metrics(sample: dict[str, float]) -> dict[str, list[str]]:
    """Map each role to the series names that matched it, so the UI can show the binding."""
    return {role: sorted(k for k in sample if pred(specmon._name(k)))
            for role, pred in BINDINGS}


def engine_metrics(url: str) -> dict:
    """Scrape the engine's Prometheus endpoint and reduce it to the panel's roles."""
    try:
        sample = specmon.fetch(url)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "values": {}, "bound": {}}
    bound = bind_metrics(sample)
    values = {role: (sum(sample[k] for k in keys) if keys else None)
              for role, keys in bound.items()}
    if values.get("prefix_queries") and values.get("prefix_hits") is not None:
        values["prefix_hit_rate_lifetime"] = values["prefix_hits"] / values["prefix_queries"]
    return {"values": values, "bound": bound, "n_series": len(sample)}


DEV_FIXTURE = "23028, 15628, 97, 271.44, 300.00, 71\n"
PROC_FIXTURE = "3412, 15320\n3999, 308\n"
LOG_FIXTURE = """
INFO 09-01 10:22:14 gpu_worker.py:284] Available KV cache memory: 9.42 GiB
INFO 09-01 10:22:15 kv_cache_utils.py:864] GPU KV cache size: 68,592 tokens
INFO 09-01 10:22:15 kv_cache_utils.py:868] Maximum concurrency for 16,384 tokens per request: 4.19x
"""
# Captured from vLLM 0.27.1 on the box, 2026-09-02. Includes the decoys that broke the
# first binding: _created timestamps, external_ variants and the _by_reason breakdown.
PROM_FIXTURE = """# HELP whatever
vllm:num_requests_running{engine="0",model_name="q"} 3.0
vllm:num_requests_waiting{engine="0",model_name="q"} 7.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="q",reason="capacity"} 5.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="q",reason="deferred"} 2.0
vllm:kv_cache_usage_perc{engine="0",model_name="q"} 0.412
vllm:prefix_cache_queries_total{engine="0",model_name="q"} 1000.0
vllm:prefix_cache_queries_created{engine="0",model_name="q"} 1788319007.8163457
vllm:prefix_cache_hits_total{engine="0",model_name="q"} 830.0
vllm:prefix_cache_hits_created{engine="0",model_name="q"} 1788319007.8163683
vllm:external_prefix_cache_queries_total{engine="0",model_name="q"} 0.0
vllm:external_prefix_cache_queries_created{engine="0",model_name="q"} 1788319007.8163999
vllm:external_prefix_cache_hits_total{engine="0",model_name="q"} 0.0
"""


def selftest() -> int:
    fails: list[str] = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"  FAIL {name}: got {got!r}, want {want!r}")

    g = parse_gpu(DEV_FIXTURE, PROC_FIXTURE)
    chk("device count", len(g["devices"]), 1)
    chk("memory_used", g["devices"][0]["memory_used"], 15628.0)
    chk("power_draw", g["devices"][0]["power_draw"], 271.44)
    chk("process count", len(g["processes"]), 2)
    chk("proc0 mib", g["processes"][0]["used_mib"], 15320.0)
    chk("garbage device line ignored", len(parse_gpu("nope\n", "")["devices"]), 0)

    kv = kv_from_log(LOG_FIXTURE)
    chk("kv_tokens", kv["kv_tokens"], 68592)
    chk("kv_gib", kv["kv_gib"], 9.42)
    chk("max_concurrency", kv["max_concurrency"], 16384)
    chk("empty log yields nothing", kv_from_log(""), {})

    sample = specmon.parse_prom(PROM_FIXTURE)
    b = bind_metrics(sample)
    chk("running bound once", len(b["running"]), 1)
    chk("waiting bound once", len(b["waiting"]), 1)
    chk("kv_usage bound once", len(b["kv_usage"]), 1)
    # the prefix-cache gauge must not be captured by the kv_usage role
    if any("prefix" in k for k in b["kv_usage"]):
        fails.append("  FAIL kv_usage bound a prefix-cache series")
    chk("prefix_hits bound once", len(b["prefix_hits"]), 1)
    chk("no series binds two roles",
        max(sum(k in keys for keys in b.values()) for k in sample), 1)

    # each role must bind exactly one series against the REAL name set, not merely
    # "no series binds two roles" -- the first version passed that and still summed a
    # timestamp into a counter.
    for role in ("running", "waiting", "kv_usage", "prefix_queries", "prefix_hits"):
        if len(b[role]) != 1:
            fails.append(f"  FAIL {role} bound {len(b[role])} series, want exactly 1: {b[role]}")
    vals = {r: (sum(sample[k] for k in ks) if ks else None) for r, ks in b.items()}
    chk("waiting is the total, not total plus its breakdown", vals["waiting"], 7.0)
    chk("prefix_queries excludes the _created timestamp", vals["prefix_queries"], 1000.0)
    chk("prefix_hits excludes the _created timestamp", vals["prefix_hits"], 830.0)
    if vals["prefix_queries"] and vals["prefix_queries"] > 1e9:
        fails.append("  FAIL prefix_queries looks like a unix timestamp")

    empty = bind_metrics({})
    chk("empty scrape binds nothing", all(v == [] for v in empty.values()), True)

    # served_model must fail soft: a wrong id is a 404 on every request (runbook section 2)
    sm = served_model("http://127.0.0.1:1")
    chk("served_model unreachable -> id None", sm["id"], None)
    chk("served_model unreachable -> error set", bool(sm["error"]), True)

    print("\n".join(fails) if fails else "labbench/probes.py selftest: all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest())
