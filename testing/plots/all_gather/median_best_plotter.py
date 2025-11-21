import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python median_best_plotter.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
df = pd.read_csv(filename)

# sanity check: every result must be correct
if not df['is_correct'].eq(1).all():
    bad = df.loc[df['is_correct'] != 1, ['nprocs','send_count','algorithm_name','k','time']]
    raise RuntimeError(
        f"Found {len(bad)} incorrect measurement(s):\n"
        + bad.drop_duplicates().to_string(index=False)
    )

# Build combined label (include k when nonzero)
df['algorithm'] = df.apply(
    lambda r: f"{r['algorithm_name']} (k={int(r['k'])})" if r['k'] > 0 else r['algorithm_name'],
    axis=1
)

# ---------------- Aggregate medians ----------------
med = (
    df.groupby(['nprocs', 'send_count', 'algorithm'])['time']
      .mean()
      .reset_index()
)

# To wide: index=(nprocs, send_count), columns=algorithm, values=median time
wide = med.pivot(index=['nprocs','send_count'], columns='algorithm', values='time')

# Baseline for speedup annotation (must exist in data)
BASELINE = 'allgather_standard'
if BASELINE not in wide.columns:
    raise RuntimeError(f"Baseline '{BASELINE}' not found in data. Available algorithms:\n{list(wide.columns)}")

# ---------------- Winners & speedup ----------------
# Global best (lowest time) per (nprocs, send_count)
best_algo_series = wide.idxmin(axis=1)          # index: (nprocs, send_count) -> algo
best_algo = best_algo_series.unstack(level=-1)  # rows: nprocs, cols: send_count

# Best-of-mine vs baseline (exclude baseline when picking "mine")
my_cols = [c for c in wide.columns if c != BASELINE]
if not my_cols:
    raise RuntimeError("No non-baseline algorithms found to compare against the baseline.")

baseline_series = wide[BASELINE]
my_best_series = wide[my_cols].min(axis=1)
speedup_series = baseline_series / my_best_series   # >1× means you beat baseline
speedup = speedup_series.unstack(level=-1)

# ---------------- Color mapping (winners only) ----------------
# Keep overall algorithm ordering stable, then filter to winners
algos = sorted(med['algorithm'].unique())
winners_set = set(best_algo_series.dropna().unique())
winners = [a for a in algos if a in winners_set]

if not winners:
    raise RuntimeError("No winners detected to display. Check your input data.")

# Choose a qualitative colormap sized to the number of winners
cmap_name = 'tab10' if len(winners) <= 10 else 'tab20'
cmap = plt.get_cmap(cmap_name, len(winners))

# Map each winner to a compact color index
winner_code = {a: i for i, a in enumerate(winners)}

# Rebuild the code matrix for pcolormesh using only winner indices
# (best_algo contains exactly winners' names, as it is argmin over 'wide')
code_matrix = best_algo.replace(winner_code)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(12, 8))

mesh = ax.pcolormesh(
    code_matrix.values,
    cmap=cmap,
    edgecolors='black',
    linewidth=0.5,
    shading='flat'
)

ax.set_aspect('equal')

# Ticks centered in cells
nprocs_vals = code_matrix.index.tolist()
send_ct_vals = code_matrix.columns.tolist()
ax.set_yticks(np.arange(len(nprocs_vals)) + 0.5)
ax.set_yticklabels(nprocs_vals)
ax.set_xticks(np.arange(len(send_ct_vals)) + 0.5)
ax.set_xticklabels(send_ct_vals, rotation=45, ha='right')

ax.set_xlabel('Send Count')
ax.set_ylabel('Number of Processes')
ax.set_title('Best Allgather Algorithm by nprocs & send_count (MEDIAN TIME)')

# Legend with only winners, each with a distinct color
patches = [
    mpatches.Patch(facecolor=cmap(winner_code[a]), edgecolor='black', label=a)
    for a in winners
]
ax.legend(
    handles=patches,
    title='Best Algorithm',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.0
)

# ---------------- Annotations (speedup) ----------------
def text_color_for_rgba(rgba):
    r, g, b, _ = rgba
    lum = 0.299*r + 0.587*g + 0.114*b
    return 'black' if lum > 0.6 else 'white'

for i, npv in enumerate(nprocs_vals):
    for j, sc in enumerate(send_ct_vals):
        # speedup value may be NaN if your algos missing for that cell
        val = speedup.get(sc, pd.Series(dtype=float)).get(npv, np.nan)
        txt = f'{val:.2f}×' if pd.notna(val) else '—'

        # background color based on winning algo in this cell
        algo_name = best_algo.get(sc, pd.Series(dtype=object)).get(npv, None)
        if algo_name in winner_code:
            rgba = cmap(winner_code[algo_name])
        else:
            rgba = (1, 1, 1, 1)  # fallback (shouldn't happen)
        ax.text(j + 0.5, i + 0.5, txt,
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color=text_color_for_rgba(rgba))

plt.tight_layout()
plt.show()