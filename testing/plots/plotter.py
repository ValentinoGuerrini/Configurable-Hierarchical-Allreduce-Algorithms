import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import numpy as np

# Load your data
df = pd.read_csv('../results.csv')

# sanity check: every result must be correct
if not df['is_correct'].eq(1).all():
    # grab the bad rows for reporting
    bad = df.loc[df['is_correct'] != 1, ['nprocs','send_count','algorithm_name','k','time']]
    raise RuntimeError(
        f"Found {len(bad)} incorrect measurement(s):\n"
        + bad.drop_duplicates().to_string(index=False)
    )

# Build a combined label (include k when nonzero)
df['algorithm'] = df.apply(
    lambda r: f"{r['algorithm_name']} (k={int(r['k'])})" if r['k'] > 0 else r['algorithm_name'],
    axis=1
)

# … the rest of your script …

# Compute median time for each (nprocs, send_count, algorithm)
med = (
    df
    .groupby(['nprocs', 'send_count', 'algorithm'])['time']
    .median()
    .reset_index()
)

# Pivot into a 3D table: index=nprocs, columns=send_count, values=median-time per algorithm
# Then for each (nprocs, send_count) pick the algorithm with min time
pivot = med.pivot_table(index='nprocs', columns='send_count', values='time', aggfunc=list)

# Instead, simpler: unstack algorithm
# Create a 3‐D DataFrame: index=(nprocs, send_count), columns=algorithm
wide = med.pivot(index=['nprocs','send_count'], columns='algorithm', values='time')

# For each cell, find the algorithm with minimum median time
best_algo = wide.idxmin(axis=1).unstack(level=-1)

# Now map each algorithm name to an integer code
algos = sorted(med['algorithm'].unique())
code = {a:i for i,a in enumerate(algos)}

# Build an integer matrix same shape as best_algo
code_matrix = best_algo.replace(code)

# Choose a discrete colormap
cmap = plt.get_cmap('tab20', len(algos))



fig, ax = plt.subplots(figsize=(12, 8))

# Use pcolormesh so we can draw edges
# pcolormesh wants the corner coordinates, so we pass code_matrix.values 
# and let it infer X=0..M, Y=0..N
mesh = ax.pcolormesh(
    code_matrix.values,
    cmap=cmap,
    edgecolors='black',    # draw black lines between cells
    linewidth=0.5,         # thickness of grid lines
    shading='flat'         # so each value is a uniform square
)

# Make squares truly square
ax.set_aspect('equal')

# Ticks in the center of each cell:
nprocs = code_matrix.index.tolist()
send_ct = code_matrix.columns.tolist()
ax.set_yticks(np.arange(len(nprocs)) + 0.5)
ax.set_yticklabels(nprocs)
ax.set_xticks(np.arange(len(send_ct)) + 0.5)
ax.set_xticklabels(send_ct, rotation=45, ha='right')

ax.set_xlabel('Send Count')
ax.set_ylabel('Number of Processes')
ax.set_title('Best Allreduce Algorithm by nprocs & send_count')

# Build a legend of colored squares
patches = [
    mpatches.Patch(facecolor=cmap(code[a]), edgecolor='black', label=a)
    for a in algos
]
ax.legend(
    handles=patches,
    title='Best Algorithm',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.0
)

plt.tight_layout()
plt.show()