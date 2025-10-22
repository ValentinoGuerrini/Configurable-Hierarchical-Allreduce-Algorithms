import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys

BASELINE = 'MPI_Reduce_scatter_block'

if len(sys.argv) < 2:
    print("Usage: python median_best_plotter.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
df = pd.read_csv(filename)

# --- keep: normalize send_count by number of processes ---
df['send_count'] = df['send_count'] / df['nprocs']

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
      .median()
      .reset_index()
)

# To wide: index=(nprocs, send_count), columns=algorithm, values=median time
wide = med.pivot(index=['nprocs','send_count'], columns='algorithm', values='time')

# Baseline for speedup annotation (must exist in data)
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

# ---------------- NEW: coloring uses best non-baseline if baseline wins ----------------
# For each cell: if the overall winner is BASELINE, color with the fastest non-baseline algo
my_best_algo_series = wide[my_cols].idxmin(axis=1)    # may be NaN if missing
color_algo_series = best_algo_series.copy()
mask_baseline_wins = (best_algo_series == BASELINE)
# Replace only where we actually have a non-baseline candidate
color_algo_series.loc[mask_baseline_wins & my_best_algo_series.notna()] = my_best_algo_series.loc[
    mask_baseline_wins & my_best_algo_series.notna()
]
color_algo = color_algo_series.unstack(level=-1)

# ---------------- Color mapping (legend shows used colors) ----------------
algos = sorted(med['algorithm'].unique())
used_set = set(color_algo_series.dropna().unique())

# Prefer to exclude baseline from legend if coloring avoided it; if baseline is the only thing left, include it.
non_baseline_used = [a for a in algos if a in used_set and a != BASELINE]
winners = non_baseline_used if non_baseline_used else [BASELINE]

# Choose a qualitative colormap sized to the number of winners
cmap_name = 'tab10' if len(winners) <= 10 else 'tab20'
cmap = plt.get_cmap(cmap_name, len(winners))

# Map each displayed algo to a compact color index
winner_code = {a: i for i, a in enumerate(winners)}

# Build the code matrix for pcolormesh using the "coloring" algorithm (not necessarily the true winner)
code_matrix = color_algo.replace(winner_code)

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

ax.set_xlabel('Recv Count')
ax.set_ylabel('Number of Processes')
ax.set_title('Best Allreduce Algorithm by nprocs & recv_count (MEDIAN TIME)\n(color shows fastest non-baseline when baseline wins)')

# Legend with only displayed algorithms
patches = [
    mpatches.Patch(facecolor=cmap(winner_code[a]), edgecolor='black', label=a)
    for a in winners
]
ax.legend(
    handles=patches,
    title='Algorithm (cell color)',
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

        # background color based on the "coloring" algo in this cell
        algo_name_for_color = color_algo.get(sc, pd.Series(dtype=object)).get(npv, None)
        if algo_name_for_color in winner_code:
            rgba = cmap(winner_code[algo_name_for_color])
        else:
            rgba = (1, 1, 1, 1)  # fallback (e.g., no non-baseline available)
        ax.text(j + 0.5, i + 0.5, txt,
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color=text_color_for_rgba(rgba))

plt.tight_layout()
plt.show()