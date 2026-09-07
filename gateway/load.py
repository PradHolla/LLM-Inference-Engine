"""
load.py -- live engine load, cached so the request path never pays a scrape.
A hook for Phase 7's thinking budget. Reads, decides nothing.

  python -m gateway.load --selftest
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import httpx

from labbench import probes

# probes already inserts tools/ on sys.path and imports specmon; reuse that rather
# than repeating the path hack, which a detached process with another cwd would break.
specmon = probes.specmon

METRICS_URL = "http://localhost:8000"
TTL_MS = 1000.0


@dataclass
class LoadSample:
    """One reading. `stale` means the scrape failed and these are the last good values."""
    running: float | None = None
    waiting: float | None = None
    kv_usage: float | None = None
    age_ms: float = 0.0
    stale: bool = False
    error: str | None = None
    n_series: int = 0


def reduce_sample(raw: dict[str, float]) -> tuple[LoadSample, dict]:
    """Bind Prometheus series to roles using probes' bindings, not a second copy."""
    bound = probes.bind_metrics(raw)
    val = {role: (sum(raw[k] for k in keys) if keys else None) for role, keys in bound.items()}
    return LoadSample(running=val.get("running"), waiting=val.get("waiting"),
                      kv_usage=val.get("kv_usage"), n_series=len(raw)), bound


class LoadSensor:
    """Polls the engine at most once per TTL. Never raises into the request path."""

    def __init__(self, url: str = METRICS_URL, ttl_ms: float = TTL_MS):
        self.url = url.rstrip("/")
        self.ttl_ms = ttl_ms
        self.last: LoadSample | None = None
        self.t_last = 0.0

    def _fresh(self, now: float) -> bool:
        return self.last is not None and (now - self.t_last) * 1e3 < self.ttl_ms

    async def sample(self, client: httpx.AsyncClient) -> LoadSample:
        """Cached reading. On failure returns the last good one, marked stale."""
        now = time.perf_counter()
        if self._fresh(now):
            s = LoadSample(**{**self.last.__dict__})
            s.age_ms = (now - self.t_last) * 1e3
            return s
        try:
            r = await client.get(f"{self.url}/metrics", timeout=2.0)
            r.raise_for_status()
            s, _ = reduce_sample(specmon.parse_prom(r.text))
        except Exception as e:
            if self.last is not None:
                s = LoadSample(**{**self.last.__dict__})
                s.stale, s.error = True, f"{type(e).__name__}"
                s.age_ms = (now - self.t_last) * 1e3
                return s
            return LoadSample(stale=True, error=f"{type(e).__name__}")
        self.last, self.t_last = s, now
        return s


def selftest() -> int:
    """Offline. The fixture carries the decoys that broke the first binding."""
    fails = []

    raw = specmon.parse_prom(probes.PROM_FIXTURE)
    s, bound = reduce_sample(raw)
    if s.running != 3.0:
        fails.append(f"running {s.running}, expected 3.0")
    if s.waiting != 7.0:
        fails.append(f"waiting {s.waiting}, expected 7.0 -- _by_reason must not be summed in")
    if s.kv_usage != 0.412:
        fails.append(f"kv_usage {s.kv_usage}, expected 0.412")
    for role, keys in bound.items():
        if len(keys) > 1:
            fails.append(f"role {role} bound {len(keys)} series: {keys}")
    if any("_created" in k for keys in bound.values() for k in keys):
        fails.append("a _created timestamp was bound into a counter role")
    if any("external_" in k for keys in bound.values() for k in keys):
        fails.append("an external_ series was bound into a real role")

    sensor = LoadSensor(ttl_ms=50_000)
    sensor.last, sensor.t_last = LoadSample(running=9.0), time.perf_counter()

    class _Boom:
        async def get(self, *a, **k):
            raise httpx.ConnectError("refused")

    import asyncio
    got = asyncio.run(sensor.sample(_Boom()))
    if got.running != 9.0:
        fails.append("a fresh cached sample was not served")

    sensor.ttl_ms = 0.0
    got = asyncio.run(sensor.sample(_Boom()))
    if not got.stale or got.running != 9.0:
        fails.append(f"a failed scrape must serve the last good values as stale: {got}")
    if got.error is None:
        fails.append("a failed scrape must record the error")

    cold = LoadSensor(ttl_ms=0.0)
    got = asyncio.run(cold.sample(_Boom()))
    if not got.stale or got.running is not None or got.error is None:
        fails.append(f"first-ever scrape failure must be stale with no values: {got}")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else print(__doc__) or 0)
