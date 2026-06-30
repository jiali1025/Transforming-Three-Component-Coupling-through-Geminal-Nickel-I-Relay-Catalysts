# keep_other_reagent_max6.py
"""Filter rows by the number of molecules in other_reagent."""

import pandas as pd, numpy as np, shutil, tempfile
from pathlib import Path
from tqdm import tqdm

SRC  = "/sharefs/lijl/xingjie/dataset/final_data_featurizable.csv"
DST  = "/sharefs/lijl/xingjie/dataset/final_data_featurizable_max.csv"
CHUNK = 250_000                 # Note: see the surrounding code for details.

def count_mols(cell: str | float) -> int:
    """English documentation for count_mols."""
    if not isinstance(cell, str) or not cell.strip():
        return 0
    return sum(1 for s in cell.split('.') if s)

# Note: see the surrounding code for details.
TOTAL = sum(1 for _ in open(SRC, 'rb')) - 1   # Note: see the surrounding code for details.
print(f'Original rows: {TOTAL:,}')

# Note: see the surrounding code for details.
tmp   = Path(tempfile.mkstemp(suffix=".csv", dir=Path(DST).parent)[1])
first = True
kept_counter = 0

with tqdm(total=TOTAL, unit="lines") as bar:
    for chunk in pd.read_csv(SRC, chunksize=CHUNK):
        mask  = chunk["other_reagent"].apply(count_mols) <= 6
        kept  = chunk.loc[mask]
        kept_counter += len(kept)

        kept.to_csv(tmp,
                    mode='w' if first else 'a',
                    header=first,
                    index=False)
        first = False
        bar.update(len(chunk))

# Note: see the surrounding code for details.
shutil.move(tmp, DST)

# Note: see the surrounding code for details.
dropped = TOTAL - kept_counter
print('=== Summary ===')
print(f'Rows kept: {kept_counter:,}')
print(f'Rows dropped: {dropped:,}')
print(f'Output file: {DST}')
