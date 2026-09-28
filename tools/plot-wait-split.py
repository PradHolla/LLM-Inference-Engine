#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Per turn of one chat: search's share of the wait to the first word vs the model's share of the whole wait.

  uv run tools/plot-wait-split.py --app results/p6b-app.jsonl --gw results/p6b-gw.jsonl
"""
import argparse, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

ap = argparse.ArgumentParser()
ap.add_argument("--app", default="results/p6b-app.jsonl")
ap.add_argument("--gw", default="results/p6b-gw.jsonl")
ap.add_argument("--out", default="results/wait-split.png")
args = ap.parse_args()

gw = {(g["chat_id"], g["turn_index"]): g for g in load(args.gw) if g.get("purpose", "answer") == "answer"}
turns, search_share, model_share = [], [], []
for r in sorted((r for r in load(args.app) if r.get("phase") == "convo"), key=lambda r: r["turn_index"]):
    g = gw[(r["chat_id"], r["turn_index"])]
    turns.append(r["turn_index"])
    search_share.append(100 * (g["search_ms"] + g["fetch_ms"] + g["extract_ms"]) / r["ttft_ms"])
    model_share.append(100 * g["e2e_ms"] / r["e2e_ms"])
# Same quantities as results/p6b-report.txt; refuse to plot if they drift from it.
assert round(search_share[0], 1) == 75.8 and round(model_share[0], 1) == 94.2, (search_share[0], model_share[0])

BLUE, ORANGE, INK, MUTED = "#1f4e79", "#c1440e", "#222222", "#666666"
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
ax.axhspan(min(model_share), max(model_share), color=BLUE, alpha=0.07, lw=0)
ax.axhspan(min(search_share), max(search_share), color=ORANGE, alpha=0.07, lw=0)
ax.plot(turns, model_share, "o", ms=8, color=BLUE)
ax.plot(turns, search_share, "o", ms=8, color=ORANGE)

ax.text(10.35, sum(model_share) / len(model_share),
        f"the model's share of the\nwait for the whole answer\n{min(model_share):.0f}-{max(model_share):.0f}%",
        color=BLUE, fontsize=11.5, fontweight="bold", va="center")
ax.text(10.35, sum(search_share) / len(search_share),
        f"web search's share of the\nwait for the first word\n{min(search_share):.0f}-{max(search_share):.0f}%",
        color=ORANGE, fontsize=11.5, fontweight="bold", va="center")

ax.set_xticks(turns); ax.set_xlim(0.6, 10.3); ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
ax.set_xlabel("turn of the conversation", fontsize=12)
ax.set_ylabel("share of the wait", fontsize=12)
ax.set_title("Same chat, two clocks\nWhat you should optimise depends on which wait you measure",
             fontsize=14, fontweight="bold", loc="left", pad=14, color=INK)
ax.grid(axis="y", alpha=0.25, lw=0.7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.01, 0.015, "Qwen3-8B fp8 weights and KV cache on one A10G 24GB, vLLM 0.27.1, 16k context. One 10-turn chat, "
         "one user, live Brave search, thinking capped at 128 tokens.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 0.80, 1))
fig.savefig(args.out, facecolor="#ffffff")
print(f"wrote {args.out}")
