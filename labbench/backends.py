"""
backends.py -- start exactly one inference server and prove the others are gone.
The backend is chosen from an enum, never from request text, and the GPU is polled for
released memory before any launch. See NOTES/code-notes.md for why both matter.
"""
from __future__ import annotations

import ast
import asyncio
import re
import shlex
import sys
import time
from dataclasses import dataclass, field

from . import probes

WORKDIR = "/opt/llm"
VENV = "/opt/llm/.venv"
VENV_VLLM = "/opt/llm/.venv-vllm"
HF_CACHE = "/opt/llm/hf-cache"
MODEL = "Qwen/Qwen3-8B"
MODEL_W4A16 = "RedHatAI/Qwen3-8B-quantized.w4a16"
EAGLE3 = "RedHatAI/Qwen3-8B-speculator.eagle3"

UNITS = {"vllm": "vllm", "baseline": "llm-baseline", "engine": "llm-engine"}
QUANTS = ("bf16", "fp8", "int4")

STOP_TIMEOUT = 60.0
GPU_FREE_TIMEOUT = 90.0
READY_TIMEOUT = 300.0
GPU_FREE_MIB = 1024.0          # a loaded 8B is 15,000+ MiB; 1 GiB means the big one is gone

BASE_ENV = [f"--setenv=HF_HOME={HF_CACHE}", "--setenv=HF_HUB_OFFLINE=1",
            "--setenv=PYTHONUNBUFFERED=1"]


def build_argv(backend: str, quant: str = "fp8", max_model_len: int = 16384,
               spec: bool = False, max_batch: int = 8) -> list[str]:
    """Full systemd-run argv for one backend. Raises on anything not in the enum."""
    if backend not in UNITS:
        raise ValueError(f"unknown backend {backend!r}")
    if quant not in QUANTS:
        raise ValueError(f"unknown quantization {quant!r}")
    head = ["sudo", "systemd-run", f"--unit={UNITS[backend]}", "--collect",
            f"--working-directory={WORKDIR}", *BASE_ENV]

    if backend == "baseline":
        return head + [f"{VENV}/bin/python", "-m", "uvicorn", "baseline.server:app",
                       "--host", "0.0.0.0", "--port", "8000"]
    if backend == "engine":
        return head + [f"{VENV}/bin/python", "-m", "engine.server",
                       "--max-batch", str(max_batch), "--host", "0.0.0.0", "--port", "8000"]

    # vLLM needs PATH for FlashInfer's ninja, and expandable_segments for +29% concurrency.
    model = MODEL_W4A16 if quant == "int4" else MODEL
    argv = head + ["--setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                   f"--setenv=PATH={VENV_VLLM}/bin:/usr/local/bin:/usr/bin:/bin",
                   f"{VENV_VLLM}/bin/python", "-m", "vllm.entrypoints.openai.api_server",
                   "--model", model, "--max-model-len", str(max_model_len),
                   "--host", "0.0.0.0", "--port", "8000"]
    if quant == "fp8":
        argv += ["--quantization", "fp8"]
    if spec:
        argv += ["--speculative-config",
                 f'{{"model":"{EAGLE3}","method":"eagle3","num_speculative_tokens":2}}']
    return argv


@dataclass
class State:
    active: str | None = None
    status: str = "stopped"           # stopped | switching | ready | failed
    stage: str | None = None
    t_stage: float = 0.0
    error: str | None = None
    quant: str = "fp8"
    config: dict = field(default_factory=dict)
    launch_cmd: str = ""

    def snapshot(self) -> dict:
        return {"backend": {"active": self.active, "status": self.status,
                            "stage": self.stage, "error": self.error,
                            "elapsed_s": round(time.monotonic() - self.t_stage, 1)
                            if self.t_stage else 0.0},
                "config": {"model": self.config.get("model", ""),
                           "quantization": self.quant,
                           "max_model_len": self.config.get("max_model_len"),
                           "kv_tokens": self.config.get("kv_tokens"),
                           "kv_gib": self.config.get("kv_gib"),
                           "spec": self.config.get("spec"),
                           "launch_cmd": self.launch_cmd}}


async def _sh(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    return await asyncio.to_thread(probes._run, cmd, timeout)


async def _wait(pred, timeout: float, interval: float = 1.0) -> bool:
    """Poll until pred() is true. Returns whether it became true -- callers MUST branch.
    A loop that can exhaust and still read as success is incident 30."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if await pred():
            return True
        await asyncio.sleep(interval)
    return await pred()


async def _units_down() -> bool:
    for unit in UNITS.values():
        rc, out = await _sh(["systemctl", "is-active", unit])
        if out.strip() == "active":
            return False
    return True


async def _gpu_released() -> bool:
    """True once no large allocation remains. Distinct from the unit being inactive:
    launching in the gap gives vLLM a silently halved KV cache (incident 39)."""
    g = await asyncio.to_thread(probes.gpu)
    devs = g.get("devices") or []
    if not devs:
        return True
    used = devs[0].get("memory_used")
    return used is None or used < GPU_FREE_MIB


async def _healthy(port: int = 8000) -> bool:
    rc, _ = await _sh(["curl", "-sf", "-m", "3", f"http://localhost:{port}/health"], 8.0)
    return rc == 0


async def active_unit() -> str | None:
    """Which backend the GPU actually has, by asking systemd rather than remembering."""
    for name, unit in UNITS.items():
        rc, out = await _sh(["systemctl", "is-active", unit])
        if out.strip() == "active":
            return name
    return None


ARGS_RE = re.compile(r"non-default args: (\{.*?\})")
SPEC_RE = re.compile(r"speculative_config=(?:SpeculativeConfig\([^)]*\)|None)")


def config_from_journal(unit: str) -> dict:
    """The config a RUNNING server actually resolved, from its own log. Needed because
    the lab bench may not have launched it, and because a restart forgets."""
    text = probes.journal_current(unit)
    cfg = probes.kv_from_log(text)
    cfg.pop("max_concurrency", None)
    m = ARGS_RE.findall(text)
    if m:
        try:
            args = ast.literal_eval(m[-1])
        except (ValueError, SyntaxError):
            args = {}
        cfg["model"] = args.get("model", "")
        cfg["max_model_len"] = args.get("max_model_len")
        cfg["quantization"] = args.get("quantization") or "bf16"
    # Same pattern infra/vllm-launch.sh greps. `speculative_config=None` appears in the
    # log of every non-spec run, so testing for the word "speculative" always matches.
    m2 = SPEC_RE.findall(text)
    cfg["spec"] = None if (not m2 or m2[-1].endswith("None")) else m2[-1]
    return cfg


class Switcher:
    """Owns the one-at-a-time invariant. The GPU is the source of truth, not the click."""

    def __init__(self) -> None:
        self.state = State()
        self._task: asyncio.Task | None = None

    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def refresh(self) -> None:
        """Reconcile with reality, so a backend that dies on its own is noticed."""
        if self.busy():
            return
        act = await active_unit()
        if act != self.state.active:
            self.state.active = act
            self.state.status = "ready" if act else "stopped"
            self.state.stage = None
            if act is None:
                self.state.config = {}
                self.state.launch_cmd = ""
        if act and not self.state.config.get("model"):
            cfg = await asyncio.to_thread(config_from_journal, UNITS[act])
            if cfg.get("model"):
                self.state.config = cfg
                self.state.quant = cfg.get("quantization", self.state.quant)

    def request(self, backend: str, quant: str | None) -> dict:
        """Accept a switch. Rejects an unknown enum before any subprocess runs."""
        if backend not in UNITS:
            return {"accepted": False, "error": f"unknown backend {backend!r}"}
        q = quant or self.state.quant
        if q not in QUANTS:
            return {"accepted": False, "error": f"unknown quantization {q!r}"}
        if self.busy():
            return {"accepted": False, "error": "a switch is already in progress"}
        self._task = asyncio.create_task(self._switch(backend, q))
        return {"accepted": True, "error": None}

    def _stage(self, name: str) -> None:
        self.state.stage = name
        self.state.t_stage = time.monotonic()

    async def _fail(self, msg: str) -> None:
        self.state.status, self.state.error = "failed", msg
        self.state.stage, self.state.active = None, await active_unit()

    async def _switch(self, backend: str, quant: str) -> None:
        st = self.state
        st.status, st.error, st.quant = "switching", None, quant
        st.launch_cmd = ""
        # We chose these, so show them immediately. Only the KV figures need the log.
        st.config = {"model": MODEL_W4A16 if quant == "int4" else MODEL,
                     "quantization": quant, "max_model_len": 16384,
                     "kv_tokens": None, "kv_gib": None, "spec": None}
        try:
            argv = build_argv(backend, quant)
        except ValueError as e:
            return await self._fail(str(e))
        st.launch_cmd = shlex.join(argv)

        self._stage("stopping")
        for unit in UNITS.values():
            await _sh(["sudo", "systemctl", "stop", unit], 30.0)
            await _sh(["sudo", "systemctl", "reset-failed", unit], 10.0)
        if not await _wait(_units_down, STOP_TIMEOUT):
            return await self._fail("units still active after stop")

        self._stage("waiting for GPU release")
        if not await _wait(_gpu_released, GPU_FREE_TIMEOUT):
            return await self._fail("GPU memory not released; refusing to launch")

        self._stage("launching")
        rc, out = await _sh(argv, 60.0)
        if rc != 0:
            return await self._fail(f"systemd-run failed: {out.strip()[:400]}")

        self._stage("loading weights")
        unit = UNITS[backend]

        async def ready_or_dead() -> bool:
            rc2, act = await _sh(["systemctl", "is-active", unit])
            if act.strip() in ("failed", "inactive"):
                raise RuntimeError(f"unit {unit} went {act.strip()} during startup")
            return await _healthy()

        try:
            ok = await _wait(ready_or_dead, READY_TIMEOUT, interval=2.0)
        except RuntimeError as e:
            return await self._fail(str(e))
        if not ok:
            return await self._fail(f"not healthy within {READY_TIMEOUT:.0f}s")

        self._stage("reading resolved config")
        text = await asyncio.to_thread(probes.journal_current, unit)
        cfg = probes.kv_from_log(text)
        cfg.pop("max_concurrency", None)
        cfg["model"] = MODEL_W4A16 if quant == "int4" else MODEL
        cfg["quantization"] = quant
        cfg["max_model_len"] = 16384
        cfg["spec"] = "eagle3" if "--speculative-config" in st.launch_cmd else None
        st.config = cfg
        st.active, st.status, st.stage, st.error = backend, "ready", None, None


def selftest() -> int:
    fails: list[str] = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"  FAIL {name}: got {got!r}, want {want!r}")

    v = build_argv("vllm", "fp8")
    chk("vllm unit", "--unit=vllm" in v, True)
    chk("vllm fp8 flag", v[v.index("--quantization") + 1], "fp8")
    chk("vllm base model", v[v.index("--model") + 1], MODEL)
    chk("vllm expandable segments",
        any("expandable_segments:True" in a for a in v), True)
    chk("vllm PATH set", any(a.startswith("--setenv=PATH=") for a in v), True)
    chk("vllm no spec by default", "--speculative-config" in v, False)

    chk("int4 uses the w4a16 checkpoint",
        build_argv("vllm", "int4")[build_argv("vllm", "int4").index("--model") + 1],
        MODEL_W4A16)
    chk("int4 carries no --quantization flag", "--quantization" in build_argv("vllm", "int4"), False)
    chk("bf16 carries no --quantization flag", "--quantization" in build_argv("vllm", "bf16"), False)
    chk("spec adds the eagle config", "--speculative-config" in build_argv("vllm", "fp8", spec=True), True)

    b = build_argv("baseline")
    chk("baseline unit", "--unit=llm-baseline" in b, True)
    chk("baseline venv", any(a == f"{VENV}/bin/python" for a in b), True)
    chk("baseline is not vllm venv", any(VENV_VLLM in a for a in b), False)
    e = build_argv("engine")
    chk("engine unit", "--unit=llm-engine" in e, True)
    chk("engine max-batch", e[e.index("--max-batch") + 1], "8")

    # absolute paths only: systemd-run runs as root, so ~ and $HOME expand wrong (incident 38)
    for name in UNITS:
        bad = [a for a in build_argv(name) if a.startswith("~") or "$HOME" in a]
        chk(f"{name} has no home-relative path", bad, [])

    for bad_call, label in (((("nope",), {}), "unknown backend"),
                            ((("vllm",), {"quant": "fp4"}), "unknown quantization")):
        args, kw = bad_call
        try:
            build_argv(*args, **kw)
            fails.append(f"  FAIL {label} was accepted")
        except ValueError:
            pass

    async def _t():
        calls = {"n": 0}

        async def never():
            calls["n"] += 1
            return False

        t0 = time.monotonic()
        got = await _wait(never, 0.3, interval=0.1)
        return got, calls["n"], time.monotonic() - t0

    got, ncalls, dt = asyncio.run(_t())
    chk("_wait returns False on timeout", got, False)
    if ncalls < 2:
        fails.append(f"  FAIL _wait polled only {ncalls} times")
    if dt > 2.0:
        fails.append(f"  FAIL _wait overran its timeout: {dt:.2f}s")

    j = "INFO speculative_config=None, tokenizer=x\nnon-default args: {'model': 'Qwen/Qwen3-8B', 'max_model_len': 16384}"
    chk("spec None is read as None", SPEC_RE.findall(j)[-1].endswith("None"), True)
    j2 = "speculative_config=SpeculativeConfig(method='eagle3', num_spec=2), x=1"
    chk("a real spec config is not None", SPEC_RE.findall(j2)[-1].endswith("None"), False)
    chk("the word alone does not imply spec", bool(SPEC_RE.findall("speculative decoding is nice")), False)

    s = State(active="vllm", status="ready", quant="fp8",
              config={"kv_tokens": 68592, "kv_gib": 9.42})
    snap = s.snapshot()
    chk("snapshot kv_tokens", snap["config"]["kv_tokens"], 68592)
    chk("snapshot active", snap["backend"]["active"], "vllm")
    chk("snapshot missing key is None", snap["config"]["max_model_len"], None)

    print("\n".join(fails) if fails else "labbench/backends.py selftest: all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest())
