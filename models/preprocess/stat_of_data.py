import pandas as pd
import numpy as np

CSV = "/sharefs/lijl/xingjie/dataset/final_data_featurizable.csv"   # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
df = pd.read_csv(CSV, usecols=["other_reagent"])

# Note: see the surrounding code for details.
counts = (
    df["other_reagent"]
      .fillna("")                                  # NaN -> empty string.
      .str.split(".")                              # Note: see the surrounding code for details.
      .apply(lambda lst: len([s for s in lst if s]))   # Note: see the surrounding code for details.
)

# Note: see the surrounding code for details.
freq = counts.value_counts().sort_index()

print('=== Summary ===')
for k, v in freq.items():
    print(f'Status: {k:2d} {v:,}')

# Note: see the surrounding code for details.
print('=== Summary ===')
print(f'Summary statistic: {len(counts):,}')
print(f'Summary statistic: {counts.max()}')
print(f'Summary statistic: {counts.mean():.2f}')
print(f'Summary statistic: {counts.median()}')
for p in [75, 90, 95, 99]:
    print(f'Summary statistic: {p} {np.percentile(counts, p)}')
