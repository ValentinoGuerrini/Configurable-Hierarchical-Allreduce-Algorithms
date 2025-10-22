#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fmt_x(x: float) -> str:
    """Compact label for send_count/nprocs values."""
    if np.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6g}"

def main():
    parser = argparse.ArgumentParser(
        description="One plot per nprocs: median time vs (send_count/nprocs) for all algorithms (k-variants separate)."
    )
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--outdir", default="plots_median_per_algo", help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots interactively")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Sanity check
    if "is_correct" in df.columns and not df["is_correct"].eq(1).all():
        bad = df.loc[df["is_correct"] != 1, ["nprocs","send_count","algorithm_name","k","time"]]
        raise RuntimeError(
            f"Found {len(bad)} incorrect measurement(s):\n"
            + bad.drop_duplicates().to_string(index=False)
        )

    need = {"nprocs","send_count","algorithm_name","k","time"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns: {sorted(miss)}")

    # Distinguish algorithm variants by k
    df["algorithm"] = df.apply(
        lambda r: f"{r['algorithm_name']} (k={int(r['k'])})"
        if pd.notna(r["k"]) and int(r["k"]) != 0 else str(r["algorithm_name"]),
        axis=1
    )

    # exact normalized x
    df["send_count_norm"] = df["send_count"] / df["nprocs"]

    # aggregate medians
    med = (
        df.groupby(["nprocs","send_count_norm","algorithm"], as_index=False)["time"]
          .median()
          .rename(columns={"time":"median_time"})
    )

    os.makedirs(args.outdir, exist_ok=True)

    for nprocs_val in sorted(med["nprocs"].unique()):
        sub = med[med["nprocs"] == nprocs_val].copy().sort_values("send_count_norm")

        fig, ax = plt.subplots(figsize=(12, 7))

        # plot one line per algorithm
        for algo, g in sub.groupby("algorithm"):
            g = g.sort_values("send_count_norm")
            ax.plot(g["send_count_norm"].values, g["median_time"].values,
                    marker="o", label=algo)

        # log x-axis
        ax.set_xscale("log")
        ax.set_xlabel("Send Count / nprocs (log scale)")
        ax.set_ylabel("Median Time")
        ax.set_title(f"Median Time vs (send_count / nprocs) — nprocs = {nprocs_val}")
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.4)

        # --- get unique x values and set them as ticks + labels ---
        xs = np.sort(sub["send_count_norm"].unique())
        ax.set_xticks(xs)
        ax.set_xticklabels([fmt_x(x) for x in xs], rotation=45, ha="right")

        # --- vertical lines at each x position ---
        for x in xs:
            ax.axvline(x=x, linestyle="--", color="gray", alpha=0.25, linewidth=0.8, zorder=0)

        ax.legend(title="Algorithm", fontsize=8, ncol=1, frameon=True)
        plt.tight_layout()

        out = os.path.join(args.outdir, f"median_by_algo_nprocs_{nprocs_val}.png")
        plt.savefig(out, dpi=150)
        if args.no_show:
            plt.close(fig)

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python make_median_algo_plots.py <results.csv> [--outdir DIR] [--no-show]")
        sys.exit(1)
    main()