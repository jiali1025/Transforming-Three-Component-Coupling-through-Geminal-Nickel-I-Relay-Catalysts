import pandas as pd
import numpy as np
from collections import OrderedDict   # Note: see the surrounding code for details.

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
SRC = "/sharefs/lijl/xingjie/dataset/pistachio_merged.csv"           # Note: see the surrounding code for details.
DST = "/sharefs/lijl/xingjie/dataset/pistachio_merged_clean.csv"

df = pd.read_csv(SRC)                  # Note: see the surrounding code for details.

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
def dedup_reagents(cell: str) -> str:
    """English documentation for dedup_reagents."""
    if pd.isna(cell) or not cell.strip():
        return ""
    uniq = OrderedDict()
    for comp in cell.split('.'):
        comp = comp.strip()
        if comp and comp not in uniq:
            uniq[comp] = None
    return '.'.join(uniq.keys())

df["reagents"] = df["reagents"].apply(dedup_reagents)

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
df["n_reagents"] = (df["reagents"]
                    .str.split('.')
                    .str.len()
                    .fillna(0)
                    .astype(np.uint8))

print('Status message.', df["n_reagents"].max())
print('Status message.')
print(df["n_reagents"].value_counts().sort_index().head(15))

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
df.to_csv(DST, index=False)
print(f'Output file: {DST}')