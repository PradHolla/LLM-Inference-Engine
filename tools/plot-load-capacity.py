#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Under load: answer speed per user vs reading speed, and what the wait for the first word is made of.

  uv run tools/plot-load-capacity.py --load results/p6d-load.jsonl results/p6d-load2.jsonl --gw results/p6d-gw.jsonl.gz
"""
import argparse, gzip, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

READING_TOK_S = 250 * 4 / 3 / 60   # 250 words/min at 4 tokens per 3 words (Baseten glossary)

def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[(len(xs) - 1) // 2]

ap = argparse.ArgumentParser()
ap.add_argument("--load", nargs="+", default=["results/p6d-load.jsonl", "results/p6d-load2.jsonl"])
ap.add_argument("--gw", default="results/p6d-gw.jsonl.gz")
ap.add_argument("--out", default="results/load-capacity.png")
args = ap.parse_args()

gw = {(g.get("chat_id"), g.get("turn_index")): g for g in map(json.loads, gzip.open(args.gw, "rt"))
      if g.get("purpose", "answer") == "answer"}
recs = [r for f in args.load for r in map(json.loads, open(f)) if r["status"] == "ok"]
rates = sorted({r["rate"] for r in recs})
running, speed, plan, read = [], [], [], []
for rate in rates:
    rs = [r for r in recs if r["rate"] == rate]
    running.append(sum(r["active_streams"] for r in rs) / len(rs))
    speed.append(med([r["stats"].get("decode_tok_s") for r in rs]))
    plan.append(med([r["stats"].get("plan_ms") for r in rs]) / 1000)
    read.append(med([r["stats"].get("engine_ttft_ms") for r in rs]) / 1000)
# Same numbers as NOTES/predictions.md P6D; refuse to plot if the inputs drifted.
assert round(speed[0]) == 52 and round(speed[-1]) == 16 and round(plan[-1], 1) == 2.2, (speed, plan)

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 11})
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
x = list(range(len(rates)))
labels = [f"{n:.0f}" for n in running]

for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.set_xticks(x, labels)
    ax.set_xlabel("load level: people waiting on an answer at once (mean)", fontsize=10.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

# Left: per-user answer speed against reading speed.
left.axhline(READING_TOK_S, color=INK2, lw=1.2, ls=(0, (4, 3)))
left.text(x[-1] + 0.3, READING_TOK_S + 1.5, f"how fast people read, ~{READING_TOK_S:.1f} tok/s",
          ha="right", va="bottom", color=INK2, fontsize=10)
left.fill_between(x, READING_TOK_S, speed, color=BLUE, alpha=0.10, lw=0)
left.plot(x, speed, color=BLUE, lw=2.2, solid_capstyle="round", zorder=3)
left.scatter(x, speed, s=46, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
for i, off, ha in ((0, (0, 12), "center"), (len(x) - 1, (-12, 4), "right")):
    left.annotate(f"{speed[i]:.0f} tok/s", (x[i], speed[i]), xytext=off, textcoords="offset points",
                  ha=ha, color=INK, fontsize=11, fontweight="bold")
left.set_ylim(0, 62)
left.set_ylabel("answer speed per person (tokens/s)", fontsize=10.5)
left.set_title("Answers still stream ~3x faster than anyone reads",
               loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=12)

# Right: the wait before the first word, split into its two parts.
width = 0.5
for i in x:
    right.bar(i, plan[i], width, color=ORANGE, lw=0)
    top = FancyBboxPatch((i - width / 2, plan[i] + 0.02), width, read[i] - 0.02,
                         boxstyle="round,pad=0,rounding_size=0.04", mutation_aspect=0.15,
                         color=BLUE, lw=0)
    right.add_patch(top)
total = [p + r for p, r in zip(plan, read)]
for i in (0, len(x) - 1):
    right.annotate(f"{total[i]:.1f} s", (x[i], total[i]), xytext=(0, 6), textcoords="offset points",
                   ha="center", color=INK, fontsize=11, fontweight="bold")
right.set_ylim(0, 2.75)
right.set_ylabel("wait before the first word (s)", fontsize=10.5)
right.set_title("The first word waits on a plan nobody sees",
                loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=12)
handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
right.legend(handles, ["writing the hidden plan (~50 tokens)", "reading the prompt (mostly cached)"],
             loc="upper left", frameon=False, fontsize=10, labelcolor=INK2)

fig.text(0.012, 0.02, "Qwen3-8B, fp8 weights and KV cache, vLLM 0.27.1, one A10G 24 GB. Chat app with a planner, "
         "4-turn real conversations arriving at random, thinking Auto, search off. KV cache never passed 48% full; "
         "no request ever queued.", fontsize=8.3, color=MUTED, wrap=True)
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=3)
fig.savefig(args.out, facecolor=SURFACE)
print(f"wrote {args.out}  running={labels}")
