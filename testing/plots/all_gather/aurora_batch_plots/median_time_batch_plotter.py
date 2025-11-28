import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 10

# ---------------------------------------------------------
# List of CSV files (edit these paths)
# ---------------------------------------------------------
csv_files = [
    "./results_aurora_batch_comparison_2.csv",
    "./results_aurora_batch_comparison_4.csv",
    "./results_aurora_batch_comparison_8.csv",
    "./results_aurora_batch_comparison_16.csv",
    "./results_aurora_batch_comparison_26.csv",
    "./results_aurora_batch_comparison_32.csv"
]

titles = [
    "208 Processes (2 Nodes)",
    "416 Processes (4 Nodes)",
    "832 Processes (8 Nodes)",
    "1664 Processes (16 Nodes)",
    "2704 Processes (26 Nodes)",
    "3328 Processes (32 Nodes)"
]

# ---------------------------------------------------------
# Create subplots: 3 rows × 2 columns
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
axes = axes.flatten()  # easier indexing

# ---------------------------------------------------------
# Function to plot a single CSV in a given axis
# ---------------------------------------------------------
def process_and_plot(ax, df, title):

    # median time per (send_count, k, b)
    grouped = (
        df.groupby(["send_count", "k", "b"])["time"]
          .mean()
          .reset_index()
    )

    # choose best k for each (send_count, b)
    best_k_per_b = (
        grouped.sort_values(["send_count", "b", "time"])
               .groupby(["send_count", "b"])
               .first()
               .reset_index()
    )

    send_counts = sorted(best_k_per_b["send_count"].unique())

    # find best b
    avg_times = best_k_per_b.groupby("b")["time"].median()
    best_b_value = avg_times.idxmin()

    # plot all b values
    for b_value, subdf in best_k_per_b.groupby("b"):
        subdf = subdf.sort_values("send_count")

        if b_value == best_b_value:
            lw = 3.0
            alpha = 1.0
            z = 5
        else:
            lw = 2.0
            alpha = 0.7
            z = 4

        ax.plot(
            subdf["send_count"],
            subdf["time"],
            marker="o",
            markersize=4,
            linewidth=lw,
            alpha=alpha,
            label=f"b = {b_value}",
            zorder=z
        )

    # annotate best b
    mid_sc = send_counts[len(send_counts) // 2]
    mid_time = best_k_per_b[
        (best_k_per_b["b"] == best_b_value) &
        (best_k_per_b["send_count"] == mid_sc)
    ]["time"].values[0]

    label = "$b^{*} = " + str(best_b_value) + r"\approx \sqrt{P}$"

    ax.annotate(
        label,
        xy=(mid_sc, mid_time),
        xytext=(mid_sc * 4, mid_time / 2.5),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", linewidth=1.6)
    )

    # formatting
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(send_counts)
    ax.set_xticklabels(send_counts, rotation=45)

    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("send_count")
    ax.set_ylabel("Median Time (s)")

    ax.legend(fontsize=9, frameon=False)


# ---------------------------------------------------------
# Loop through CSVs and plot each one
# ---------------------------------------------------------
for i, (csv_path, title) in enumerate(zip(csv_files, titles)):
    df = pd.read_csv(csv_path)
    df = df[df["is_correct"] == 1]   # filter correctness
    process_and_plot(axes[i], df, title)

plt.tight_layout()
plt.show()