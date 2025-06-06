#!/usr/bin/env python3
"""
check_and_visualize_recv.py

Reads “all_buffers.txt” (or similar) containing both initial send buffers and final recv buffers.
Parses the file to extract:
  - r, b parameters
  - Initial send buffers (“SendRank” lines) ─ not directly used for correctness check
  - Final recv buffers (“Rank” lines)

Then computes the expected recv-buffer values under MPI_SUM reduce_scatter:
  expected[i, k] = sum_{R=0..(P-1)} [ R*1000 + (i*count + k) ]
                 = 1000 * (0 + 1 + … + (P-1)) + P * (i*count + k)
                 = 1000 * (P*(P-1)/2)     + P * (i*count + k)

Compares the actual recv data to expected, prints a summary of mismatches (if any),
and displays three side-by-side heatmaps:
  1) Actual recv buffers
  2) Expected recv buffers
  3) Difference (actual – expected) with a diverging colormap centered at zero

Usage:
    python3 check_and_visualize_recv.py all_buffers.txt
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import re

def read_buffers(filename):
    """
    Parses "all_buffers.txt" (or similar) and returns:
      - r_val, b_val: integers from the header “(r=<r>, b=<b>)”
      - send_data: 2D numpy array of shape (P, total_send_elems)  [not used here]
      - recv_data: 2D numpy array of shape (P, count)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    r_val = None
    b_val = None
    send_lines = []
    recv_lines = []

    # Regex to extract r and b from lines like "## initial send buffers (r=2, b=7)"
    header_pattern = re.compile(r'\(r\s*=\s*(\d+)\s*,\s*b\s*=\s*(\d+)\)')

    for line in lines:
        stripped = line.strip()

        # Capture r, b from the "## initial send buffers" header
        if stripped.startswith("## initial send buffers"):
            m = header_pattern.search(stripped)
            if m:
                r_val = int(m.group(1))
                b_val = int(m.group(2))

        # Collect send-buffer lines (prefix "SendRank")
        elif stripped.startswith("SendRank"):
            send_lines.append(stripped)

        # Collect recv-buffer lines (prefix "Rank" but NOT "SendRank")
        elif stripped.startswith("Rank"):
            recv_lines.append(stripped)

    if not recv_lines:
        raise RuntimeError("No lines starting with 'Rank' found in " + filename)

    # Parse send_lines into a 2D list of ints (for completeness; not strictly needed)
    send_parsed = []
    for line in send_lines:
        colon_pos = line.find(':')
        if colon_pos < 0:
            raise RuntimeError(f"Malformed SendRank line: {line}")
        nums_str = line[colon_pos+1:].strip()
        if not nums_str:
            raise RuntimeError(f"No data after ':' in SendRank line: {line}")
        row = [int(tok) for tok in nums_str.split()]
        send_parsed.append(row)

    # Parse recv_lines into a 2D list of ints
    recv_parsed = []
    for line in recv_lines:
        colon_pos = line.find(':')
        if colon_pos < 0:
            raise RuntimeError(f"Malformed Rank line: {line}")
        nums_str = line[colon_pos+1:].strip()
        if not nums_str:
            raise RuntimeError(f"No data after ':' in Rank line: {line}")
        row = [int(tok) for tok in nums_str.split()]
        recv_parsed.append(row)

    # Convert lists to NumPy arrays
    send_data = None
    if send_parsed:
        send_data = np.vstack([np.array(row, dtype=int) for row in send_parsed])
    else:
        send_data = np.empty((0, 0), dtype=int)

    recv_data = np.vstack([np.array(row, dtype=int) for row in recv_parsed])

    return r_val, b_val, send_data, recv_data

def compute_expected_recv(P, count):
    """
    Given number of processes P and recv-buffer length count,
    returns a 2D array of shape (P, count) with the expected values:
      expected[i, k] = 1000 * (P*(P-1)/2) + P*(i*count + k)
    """
    total_rank_sum = P * (P - 1) // 2  # 0 + 1 + ... + (P-1)
    base = 1000 * total_rank_sum       # sum_{R} (R*1000)
    expected = np.zeros((P, count), dtype=int)

    for i in range(P):
        for k in range(count):
            expected[i, k] = base + P * (i * count + k)
    return expected

def report_mismatches(actual, expected):
    """
    Compares actual vs. expected arrays (both shape (P, count)).
    Prints a summary: if all match, says so.
    Otherwise, lists (rank, index) pairs where mismatch occurs, along with (actual, expected).
    Returns a boolean mask array mismatches of shape (P, count) with True where mismatch.
    """
    if actual.shape != expected.shape:
        raise RuntimeError("Shape mismatch between actual and expected recv arrays.")

    P, count = actual.shape
    mismatches = (actual != expected)
    num_mismatch = np.count_nonzero(mismatches)

    if num_mismatch == 0:
        print("✔ All recv-buffer values match expected values.")
    else:
        print(f"✖ Found {num_mismatch} mismatched element(s) out of {P * count}:")
        for i in range(P):
            for k in range(count):
                if mismatches[i, k]:
                    print(f"  Rank {i}, index {k}: actual={actual[i,k]}  expected={expected[i,k]}")
    return mismatches

def plot_recv_comparison(actual, expected, mismatches, r=None, b=None):
    """
    Creates a figure with 3 side-by-side subplots:
      1) Heatmap of actual recv buffers
      2) Heatmap of expected recv buffers
      3) Heatmap of (actual - expected), with a diverging colormap centered at 0

    Overlays text annotation of each cell’s value.
    Cells where actual != expected are outlined in red on the first two plots.
    """
    P, count = actual.shape
    diff = actual - expected

    fig, axes = plt.subplots(1, 3, figsize=(4 * 3 + 2, 4))
    titles = ["Actual recv buffers", "Expected recv buffers", "Difference (Actual - Expected)"]

    vmin_act, vmax_act = np.min(actual), np.max(actual)
    vmin_exp, vmax_exp = np.min(expected), np.max(expected)
    vlim = max(abs(np.min(diff)), abs(np.max(diff)))

    # 1) Actual recv buffers
    ax0 = axes[0]
    im0 = ax0.imshow(actual, aspect='auto', interpolation='nearest', cmap='viridis',
                     vmin=vmin_act, vmax=vmax_act)
    ax0.set_title(f"{titles[0]}\n(r={r}, b={b})" if r is not None and b is not None else titles[0])
    ax0.set_xlabel("Recv-buffer index")
    ax0.set_ylabel("MPI rank")
    ax0.set_xticks(np.arange(count))
    ax0.set_yticks(np.arange(P))
    ax0.set_xticklabels([str(k) for k in range(count)])
    ax0.set_yticklabels([str(i) for i in range(P)])
    for i in range(P):
        for k in range(count):
            color = "white" if actual[i, k] < (vmin_act + vmax_act) / 2 else "black"
            ax0.text(k, i, f"{actual[i, k]}", ha="center", va="center", color=color, fontsize=8)
            if mismatches[i, k]:
                # draw red rectangle around mismatched cell
                rect = plt.Rectangle((k - 0.5, i - 0.5), 1, 1, 
                                     edgecolor="red", facecolor="none", linewidth=2)
                ax0.add_patch(rect)
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("Value")

    # 2) Expected recv buffers
    ax1 = axes[1]
    im1 = ax1.imshow(expected, aspect='auto', interpolation='nearest', cmap='viridis',
                     vmin=vmin_exp, vmax=vmax_exp)
    ax1.set_title(titles[1])
    ax1.set_xlabel("Recv-buffer index")
    ax1.set_ylabel("MPI rank")
    ax1.set_xticks(np.arange(count))
    ax1.set_yticks(np.arange(P))
    ax1.set_xticklabels([str(k) for k in range(count)])
    ax1.set_yticklabels([str(i) for i in range(P)])
    for i in range(P):
        for k in range(count):
            color = "white" if expected[i, k] < (vmin_exp + vmax_exp) / 2 else "black"
            ax1.text(k, i, f"{expected[i, k]}", ha="center", va="center", color=color, fontsize=8)
            if mismatches[i, k]:
                rect = plt.Rectangle((k - 0.5, i - 0.5), 1, 1, 
                                     edgecolor="red", facecolor="none", linewidth=2)
                ax1.add_patch(rect)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Value")

    # 3) Difference heatmap
    ax2 = axes[2]
    im2 = ax2.imshow(diff, aspect='auto', interpolation='nearest', cmap='coolwarm',
                     vmin=-vlim, vmax=+vlim)
    ax2.set_title(titles[2])
    ax2.set_xlabel("Recv-buffer index")
    ax2.set_ylabel("MPI rank")
    ax2.set_xticks(np.arange(count))
    ax2.set_yticks(np.arange(P))
    ax2.set_xticklabels([str(k) for k in range(count)])
    ax2.set_yticklabels([str(i) for i in range(P)])
    for i in range(P):
        for k in range(count):
            txt_color = "white" if abs(diff[i, k]) > vlim * 0.5 else "black"
            ax2.text(k, i, f"{diff[i, k]}", ha="center", va="center", color=txt_color, fontsize=8)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Difference")

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_and_visualize_recv.py all_buffers.txt")
        sys.exit(1)

    filename = sys.argv[1]
    r_val, b_val, send_data, recv_data = read_buffers(filename)
    P, count = recv_data.shape

    # Compute expected recv data
    expected = compute_expected_recv(P, count)

    # Compare and report mismatches
    mismatches = report_mismatches(recv_data, expected)

    # Plot actual vs expected vs difference
    plot_recv_comparison(recv_data, expected, mismatches, r=r_val, b=b_val)

if __name__ == "__main__":
    main()