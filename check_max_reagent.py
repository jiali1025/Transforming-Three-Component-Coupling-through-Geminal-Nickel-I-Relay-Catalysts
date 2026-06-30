import pandas as pd
import numpy as np

# Note: see the surrounding code for details.
csv_path = "/sharefs/lijl/xingjie/dataset/pistachio_filter.csv"
usecols   = ["reagents"]                # Note: see the surrounding code for details.
df = pd.read_csv(csv_path, usecols=usecols)

# Note: see the surrounding code for details.
def count_reagents(cell: str) -> int:
    """English documentation for count_reagents."""
    if pd.isna(cell) or not cell.strip():
        return 0
    # Note: see the surrounding code for details.
    return len(list(filter(None, cell.split('.'))))

df["n_reagents"] = df["reagents"].apply(count_reagents).astype(np.uint8)  # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
max_reagents  = int(df["n_reagents"].max())
desc          = df["n_reagents"].describe()
vc            = df["n_reagents"].value_counts().sort_index()   # Note: see the surrounding code for details.

print(f'Maximum reagents per reaction: {max_reagents}')
print('Status message.')
print(desc.to_string(), "\n")

print('Status message.')
print(vc.head(20).to_string())          # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
THRESH = 6
mask_keep = df["n_reagents"] <= THRESH
print(f'Rows retained after filtering: {mask_keep.mean():.2%} {mask_keep.sum():,}')
# df_filtered = df.loc[mask_keep].copy()