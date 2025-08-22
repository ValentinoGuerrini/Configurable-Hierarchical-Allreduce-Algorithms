import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys

if len(sys.argv) < 2:
    print("Usage: python median_plotter.py <filename>")
    sys.exit(1)

filename = sys.argv[1]


SAVE_PNG = True
PNG_PATH = "./images/"+(filename.split("_")[1]).split(".")[0]+"_median_time_by_nprocs_winners_per_subplot.png"

MIN_SEND, MAX_SEND = 8, 16384

# potenze di 2 per i tick
xticks_p2 = []
v = MIN_SEND
while v <= MAX_SEND:
    xticks_p2.append(v)
    v *= 2

# === Load & aggregate ===
df = pd.read_csv(filename)

# mediana del tempo per (nprocs, algorithm_name, k, send_count)
median_df = (
    df.groupby(["nprocs", "algorithm_name", "k", "send_count"], as_index=False)["time"]
      .median()
      .rename(columns={"time": "median_time_s"})
)

# usa solo implementazioni con k > 0
median_df = median_df[median_df["k"] > 0].copy()
median_df["median_time_ms"] = median_df["median_time_s"] * 1e3

# ordini utili
nprocs_vals = sorted(median_df["nprocs"].unique().tolist())

# layout subplot
n = len(nprocs_vals)
ncols = min(3, n) if n > 1 else 1
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(5.6*ncols, 4.2*nrows),
    sharey=True, constrained_layout=False
)

if n == 1:
    axes = [axes]
axes_flat = axes if isinstance(axes, list) else axes.flatten()

# disegno
for ax, npv in zip(axes_flat, nprocs_vals):
    sub = median_df[median_df["nprocs"] == npv].copy()

    # --- trova i vincitori (k che vincono per almeno un send_count) ---
    winners = (
        sub.loc[sub.groupby("send_count")["median_time_ms"].idxmin()]["k"]
        .unique()
        .tolist()
    )

    # traccia solo gli algoritmi vincitori
    for k_val in winners:
        g = sub[sub["k"] == k_val].sort_values("send_count")
        if g.empty:
            continue
        ax.plot(
            g["send_count"], g["median_time_ms"],
            linestyle='-', marker='o',
            linewidth=1.6, markersize=4, alpha=0.9,
            label=f"k={int(k_val)}"
        )

    ax.set_title(f"nprocs = {npv}")
    ax.set_xlabel("send_count (powers of 2)")
    ax.set_ylabel("Median time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # Tick alle potenze di 2
    present = sorted(set(int(x) for x in sub["send_count"].unique() if x in xticks_p2))
    ax.set_xticks(present if present else xticks_p2)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)

    # legend per subplot
    ax.legend(title="Winning ks", frameon=False, fontsize=8)

# nascondi assi vuoti eventuali
for ax in axes_flat[len(nprocs_vals):]:
    ax.axis("off")

plt.tight_layout()

if SAVE_PNG:
    plt.savefig(PNG_PATH, dpi=220, bbox_inches="tight")

plt.show()