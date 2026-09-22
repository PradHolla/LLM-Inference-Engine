#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Warm vs cold TTFT across a conversation, from convo-*.jsonl.

  uv run tools/plot-prefix-cache.py --warm results/convo-warm.jsonl --cold results/convo-cold.jsonl
"""
import argparse, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

ap = argparse.ArgumentParser()
ap.add_argument("--warm", default="results/convo-warm.jsonl")
ap.add_argument("--cold", default="results/convo-cold.jsonl")
ap.add_argument("--out", default="results/prefix-cache-ttft.png")
args = ap.parse_args()
w, c = load(args.warm), load(args.cold)
x = [a["usage_prompt_tokens"] for a in w]

fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
ax.plot(x, [b["ttft_ms"] for b in c], "-o", ms=4, lw=2.4, color="#c1440e", label="prefix cache defeated")
ax.plot(x, [a["ttft_ms"] for a in w], "-o", ms=4, lw=2.4, color="#1f4e79", label="prefix cache working")

ax.annotate("2,291 ms", xy=(x[-1], c[-1]["ttft_ms"]), xytext=(-70, 14),
            textcoords="offset points", color="#c1440e", fontsize=13, fontweight="bold")
ax.annotate("166 ms", xy=(x[-1], w[-1]["ttft_ms"]), xytext=(-62, 16),
            textcoords="offset points", color="#1f4e79", fontsize=13, fontweight="bold")
ax.annotate("13.8x", xy=(x[-1], 1200), xytext=(-46, 0), textcoords="offset points",
            fontsize=15, fontweight="bold", color="#333333")
ax.annotate("", xy=(x[-1], w[-1]["ttft_ms"]), xytext=(x[-1], c[-1]["ttft_ms"]),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6))

ax.set_xlabel("conversation length (prompt tokens)", fontsize=12)
ax.set_ylabel("time to first token (ms)", fontsize=12)
ax.set_title("Same conversation\nThe only difference is whether the prefix cache works",
             fontsize=14, fontweight="bold", loc="left", pad=14)
ax.legend(fontsize=12, frameon=False, loc="upper left")
ax.grid(alpha=0.25, lw=0.7); ax.set_ylim(0, 2550); ax.set_xlim(0, 8200)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.01, 0.015, "Qwen3-8B fp8 on A10G 24GB, vLLM 0.27.1, 16k context, 25 turns. "
         "Cold arm verified: prefix cache hit rate 0.000 on every turn.",
         fontsize=8.5, color="#666666")
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(args.out, facecolor="#ffffff")
print(f"wrote {args.out}")
