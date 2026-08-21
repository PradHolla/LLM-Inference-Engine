#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
roofline.py -- predict what a model SHOULD do on a given GPU, from first principles.

This is not a benchmark. Every number here comes from dividing one hardware spec by
another. That is the point: run this BEFORE you measure, write the prediction down,
then explain any gap. A wrong prediction is a specific mystery. No prediction is
just a vague feeling that something is slow.

  python tools/roofline.py --model qwen3-8b --gpu a10g
  python tools/roofline.py --model qwen3-8b --gpu a10g --dtype fp8 --context 8192
  python tools/roofline.py --hf ~/models/Qwen3-8B/config.json --params 8.2e9 --gpu a10g

UNITS: memory is GiB (2^30) throughout, because that is what nvidia-smi reports.
Bandwidth is GB/s (10^9) because that is how vendors spec it. Mixing these silently
is a classic source of ~7% errors, so they are converted explicitly, never assumed.

MODEL LIMITATIONS:
  - decode assumes a DENSE model; MoE reads only active experts per token, so every
    number here is wrong for Qwen3-30B-A3B or gpt-oss-20b
  - attention FLOPs are ignored; only the weight matmuls are counted, so prefill is
    understated at long context
  - the dequant and kernel-efficiency constants above are ESTIMATES, not measurements.
    Replace them with measured values once Phase 3/4 has real numbers
  - no modeling of chunked prefill, prefix-cache hits, or scheduler overhead
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass

GIB = 1 << 30
GB = 10**9


@dataclass
class Model:
    name: str
    layers: int
    kv_heads: int      # GQA: this is much smaller than the attention head count
    head_dim: int
    params: float
    hidden: int

    def kv_bytes_per_token(self, dtype_bytes: float) -> float:
        # 2 = one K and one V. Per layer, per token.
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes

    def weight_bytes(self, dtype_bytes: float) -> float:
        return self.params * dtype_bytes


@dataclass
class GPU:
    name: str
    vram_gib: float      # what nvidia-smi actually reports, not the marketing number
    bandwidth_gb_s: float
    bf16_tflops: float   # dense, no sparsity


# VRAM values are what nvidia-smi ACTUALLY reports on the card, not the marketing
# number and not AWS's DescribeInstanceTypes (which under-reported the A10G by
# 140 MiB -- measured 23028, API said 22888). An A10G is sold as "24 GB" and gives
# you 22.49 GiB; that 1.5 GiB gap comes straight out of your KV cache budget.
#   MEASURED on i-07d8b10bdcf39a099 (g5.2xlarge), driver 595.91.07, 2026-08-21: a10g
#   The rest are still spec-sheet values -- verify before trusting them.
GPUS = {
    "a10g":  GPU("A10G (g5.*)",   23028 / 1024, 600, 125),
    "l4":    GPU("L4 (g6.*)",     22888 / 1024, 300, 121),
    "l40s":  GPU("L40S (g6e.*)",  45776 / 1024, 864, 362),
    "t4":    GPU("T4 (g4dn.*)",   16384 / 1024, 320,  65),
    "a100":  GPU("A100 40GB",     40960 / 1024, 1555, 312),
    "h100":  GPU("H100 80GB",     81559 / 1024, 3350, 990),
}

MODELS = {
    "qwen3-4b":     Model("Qwen3-4B",      36, 8, 128,  4.0e9, 2560),
    "qwen3-8b":     Model("Qwen3-8B",      36, 8, 128, 8.190735360e9, 4096),
    "qwen3-14b":    Model("Qwen3-14B",     40, 8, 128, 14.8e9, 5120),
    "llama-3.1-8b": Model("Llama-3.1-8B",  32, 8, 128,  8.03e9, 4096),
}

DTYPES = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "awq4": 0.5, "int4": 0.5}

# Ops per parameter to unpack a quantized weight back to fp16 before the
# matmul. Ampere (sm86) has no int4 or fp8 tensor cores, so this is real
# work that bf16 never pays. Cost scales with PARAMETER COUNT, not batch --
# a weight tile is dequantized once and reused down the batch dimension --
# so as a fraction of total compute it shrinks as batch grows.
DEQUANT_OPS_PER_PARAM = {"bf16": 0, "fp16": 0, "fp8": 2, "int8": 2, "awq4": 4, "int4": 4}

# Quantized kernels (Marlin, AWQ) do not reach cuBLAS fp16 GEMM efficiency at
# large batch. THIS is what actually makes 4-bit lose to fp16 in the compute-
# bound regime -- not the dequant flops above, which become negligible.
QUANT_KERNEL_EFF = {"bf16": 1.0, "fp16": 1.0, "fp8": 0.85, "int8": 0.85, "awq4": 0.85, "int4": 0.85}

# Nobody hits peak. These are the fudge factors, isolated here so they are
# arguments rather than hidden assumptions -- and so you can tune them once you
# have measured reality and know what your stack actually achieves.
MEM_EFF = 0.65      # fraction of peak bandwidth a real kernel sustains
COMPUTE_EFF = 0.50  # fraction of peak FLOPs during prefill


def load_hf_config(path: str, params: float | None) -> Model:
    cfg = json.load(open(path))
    heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads", heads)
    hidden = cfg["hidden_size"]
    head_dim = cfg.get("head_dim") or hidden // heads
    if params is None:
        sys.exit("--params is required with --hf (a config.json does not state the "
                 "parameter count; read it off the model card)")
    return Model(cfg.get("_name_or_path", path), cfg["num_hidden_layers"],
                 kv_heads, head_dim, params, hidden)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen3-8b")
    ap.add_argument("--hf", metavar="CONFIG_JSON", help="read layers/heads from a HF config.json instead")
    ap.add_argument("--params", type=float, help="parameter count, required with --hf")
    ap.add_argument("--gpu", choices=sorted(GPUS), default="a10g")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16", help="weight dtype")
    ap.add_argument("--kv-dtype", choices=sorted(DTYPES), help="KV cache dtype (default: same as --dtype, min fp16)")
    ap.add_argument("--util", type=float, default=0.90, help="vLLM --gpu-memory-utilization")
    ap.add_argument("--overhead-gib", type=float, default=1.0, help="CUDA context + activations")
    ap.add_argument("--context", type=int, default=4096, help="tokens per request, for the capacity estimate")
    ap.add_argument("--mem-eff", type=float, default=MEM_EFF)
    ap.add_argument("--compute-eff", type=float, default=COMPUTE_EFF)
    ap.add_argument("--dequant-ops", type=float, default=None, help="override ops/param for weight unpacking")
    ap.add_argument("--kernel-eff", type=float, default=None, help="override quantized-kernel efficiency vs cuBLAS fp16")
    a = ap.parse_args()

    m = load_hf_config(a.hf, a.params) if a.hf else MODELS[a.model]
    g = GPUS[a.gpu]
    wb = DTYPES[a.dtype]
    # Quantizing WEIGHTS does not quantize the CACHE -- they are independent knobs.
    # vLLM's default `--kv-cache-dtype auto` keeps the cache at the model's compute
    # dtype (fp16/bf16) even when the weights are AWQ-4bit, so that is the default
    # here too. Assuming otherwise would overstate capacity by 2x.
    kvb = DTYPES[a.kv_dtype] if a.kv_dtype else 2

    W = m.weight_bytes(wb)
    kv_tok = m.kv_bytes_per_token(kvb)
    usable = g.vram_gib * a.util
    kv_room = usable - W / GIB - a.overhead_gib
    max_tokens = kv_room * GIB / kv_tok if kv_room > 0 else 0

    print(f"\n\033[1m{m.name}\033[0m  weights={a.dtype}  kv={('fp16' if kvb==2 else 'fp8' if kvb==1 else 'int4')}"
          f"   on   \033[1m{g.name}\033[0m")
    print(f"  {m.layers} layers · {m.kv_heads} kv-heads · head_dim {m.head_dim} · {m.params/1e9:.1f}B params")

    print("\n\033[1mMEMORY BUDGET\033[0m")
    print(f"  VRAM (reported)        {g.vram_gib:8.2f} GiB")
    print(f"  × util {a.util:<4}            {usable:8.2f} GiB")
    print(f"  - weights              {-W/GIB:8.2f} GiB   ({m.params/1e9:.1f}B × {wb} B)")
    print(f"  - overhead             {-a.overhead_gib:8.2f} GiB")
    print(f"  {'':─<24} ")
    print(f"  = KV cache room        {kv_room:8.2f} GiB")
    if kv_room <= 0:
        print("\n  \033[1;31mDOES NOT FIT.\033[0m Quantize the weights or use a bigger card.\n")
        return

    print("\n\033[1mCAPACITY\033[0m   (the number that actually limits you)")
    print(f"  KV per token           {kv_tok/1024:8.1f} KiB   (2 × {m.layers} × {m.kv_heads} × {m.head_dim} × {kvb} B)")
    print(f"  max tokens in flight   {max_tokens:8,.0f}       ← across ALL users combined")
    print(f"  ÷ {a.context} tok/request      {max_tokens/a.context:8,.1f}       concurrent requests at {a.context} ctx")

    print("\n\033[1mDECODE ROOFLINE\033[0m   (batch 1 — memory bound)")
    t_mem_ms = W / (g.bandwidth_gb_s * GB) * 1000
    print(f"  floor  ms/token        {t_mem_ms:8.2f}       = {W/GB:.1f} GB ÷ {g.bandwidth_gb_s} GB/s")
    print(f"  ceiling tok/s          {1000/t_mem_ms:8.1f}       ← nothing beats this on this card")
    print(f"  realistic tok/s        {1000/t_mem_ms*a.mem_eff:8.1f}       (at {a.mem_eff:.0%} of peak bandwidth)")

    print(f"\n\033[1mBATCH SCALING\033[0m   at {a.context} ctx — where memory-bound becomes compute-bound")
    print("  Batching works because the weights are read ONCE for the whole batch.")
    print("  It stops working when either the KV reads or the math catch up.")
    dequant_ops = a.dequant_ops if a.dequant_ops is not None else DEQUANT_OPS_PER_PARAM[a.dtype]
    kernel_eff = a.kernel_eff if a.kernel_eff is not None else QUANT_KERNEL_EFF[a.dtype]
    if dequant_ops > 0:
        print("  \033[2mQuantized compute estimates carry the dequant + kernel-efficiency model")
        print("  below and are softer than the memory numbers above.\033[0m")
    print()
    print(f"  {'batch':>6} {'bytes/step':>11} {'t_mem':>8} {'t_cmp':>8} {'ITL':>8} {'tok/s':>9}  bound by")
    print(f"  {'':->6} {'':->11} {'':->8} {'':->8} {'':->8} {'':->9}  {'':->12}")
    for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if B * a.context > max_tokens:
            print(f"  {B:>6} {'—':>11} {'—':>8} {'—':>8} {'—':>8} {'—':>9}  \033[33mout of KV cache\033[0m")
            continue
        # Per decode step the GPU reads the weights once, plus every live KV entry.
        # That second term is why the knee arrives far earlier than the raw
        # FLOPs:bytes ratio of the card suggests.
        mem = W + B * a.context * kv_tok
        t_mem = mem / (g.bandwidth_gb_s * GB * a.mem_eff)
        dq = dequant_ops * m.params
        t_cmp = (2 * m.params * B + dq) / (g.bf16_tflops * 1e12 * a.compute_eff * kernel_eff)
        t = max(t_mem, t_cmp)
        bound = "memory" if t_mem >= t_cmp else "\033[36mcompute\033[0m"
        print(f"  {B:>6} {mem/GIB:>10.1f}G {t_mem*1000:>7.1f}m {t_cmp*1000:>7.1f}m "
              f"{t*1000:>7.1f}m {B/t:>9,.0f}  {bound}")

    print(f"\n\033[1mPREFILL / TTFT\033[0m   (compute bound — the metric users actually feel)")
    print(f"  {'prompt':>8} {'FLOPs':>10} {'prefill':>10} {'+1 decode':>11} {'≈ TTFT':>9}")
    print(f"  {'':->8} {'':->10} {'':->10} {'':->11} {'':->9}")
    for n in (128, 512, 2048, 8192, 32768):
        if n > max_tokens:
            break
        fl = 2 * m.params * n
        t_pre = fl / (g.bf16_tflops * 1e12 * a.compute_eff)
        t_dec = t_mem_ms / 1000 / a.mem_eff
        print(f"  {n:>8,} {fl/1e12:>9.1f}T {t_pre*1000:>9.0f}m {t_dec*1000:>10.0f}m {(t_pre+t_dec)*1000:>8.0f}m")

    print("\n  \033[2mAll of the above is arithmetic, not measurement. Write the numbers you")
    print("  care about into NOTES/predictions.md before you run anything.\033[0m\n")


if __name__ == "__main__":
    main()
