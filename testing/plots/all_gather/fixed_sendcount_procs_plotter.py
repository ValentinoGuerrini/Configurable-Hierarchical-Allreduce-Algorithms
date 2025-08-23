import pandas as pd
import matplotlib.pyplot as plt
import math

# === Load CSV ===
CSV_PATH = "results_aurora.csv"
df = pd.read_csv(CSV_PATH)

# Keep only correct runs
df = df[df["is_correct"] == 1]

# Compute median time grouped by (nprocs, send_count, k)
median_df = (
    df.groupby(["nprocs", "send_count", "k"], as_index=False)["time"]
      .median()
      .rename(columns={"time": "median_time"})
)

def pick_indices(subset):
    """
    Return indices for best (min), second-best (2nd min), and worst (max) rows.
    Handles small cardinalities robustly.
    """
    if subset.empty:
        return None, None, None

    # Sort ascending by median_time
    asc = subset.sort_values("median_time")
    best_idx = asc.index[0]

    # Second-best if available
    second_best_idx = asc.index[1] if len(asc) > 1 else None

    # Worst (max)
    worst_idx = subset["median_time"].idxmax()

    return best_idx, second_best_idx, worst_idx


# Iterate over each nprocs value
for nprocs, group in median_df.groupby("nprocs"):
    send_counts = sorted(group["send_count"].unique())
    n_subplots = len(send_counts)

    # Grid shape: square-ish
    ncols = math.ceil(math.sqrt(n_subplots))
    nrows = math.ceil(n_subplots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows), squeeze=False)
    fig.suptitle(f"Median Execution Time vs k (nprocs={nprocs})", fontsize=14)

    for i, send_count in enumerate(send_counts):
        ax = axes[i // ncols, i % ncols]

        # Separate k=0 and k>0
        subset_all = group[group["send_count"] == send_count]
        subset_k0 = subset_all[subset_all["k"] == 0]
        subset_kpos = subset_all[subset_all["k"] != 0].sort_values("k").reset_index(drop=True)

        if subset_all.empty:
            ax.set_visible(False)
            continue

        # Plot k>0 points
        if not subset_kpos.empty:
            ax.plot(subset_kpos["k"], subset_kpos["median_time"], marker="o", linestyle="-", alpha=0.8)

            # Highlight best/second-best/worst
            best_idx, second_idx, worst_idx = pick_indices(subset_kpos)

            if best_idx is not None:
                ax.scatter(subset_kpos.loc[best_idx, "k"], subset_kpos.loc[best_idx, "median_time"],
                           s=80, color="green", zorder=3, label="Best")

            if second_idx is not None:
                ax.scatter(subset_kpos.loc[second_idx, "k"], subset_kpos.loc[second_idx, "median_time"],
                           s=80, color="yellow", edgecolors="black", zorder=3, label="Second best")

            if worst_idx is not None:
                ax.scatter(subset_kpos.loc[worst_idx, "k"], subset_kpos.loc[worst_idx, "median_time"],
                           s=80, color="red", zorder=3, label="Worst")

        # Plot horizontal line for k=0 median
        if not subset_k0.empty:
            val = subset_k0["median_time"].values[0]
            ax.axhline(y=val, color="blue", linestyle="--", label="MPICH baseline")

        ax.set_title(f"send_count = {send_count}")
        ax.set_xlabel("k")
        ax.set_ylabel("Median Time (s)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best", frameon=True)

    # Hide unused subplots
    total = nrows * ncols
    for j in range(n_subplots, total):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()