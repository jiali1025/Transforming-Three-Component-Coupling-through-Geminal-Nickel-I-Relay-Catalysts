# drop_bad_rows_single.py
"""Remove rows listed in bad_rows_global_after_fix.txt and write a cleaned CSV while preserving other columns."""

import pandas as pd, numpy as np, shutil, tempfile
from pathlib import Path
from tqdm import tqdm

# Note: see the surrounding code for details.
SRC_CSV   = "/sharefs/lijl/xingjie/dataset/reagents_split_by_catalyst_fixed.csv"
BAD_FILE  = "bad_rows_global_after_fix.txt"      # Note: see the surrounding code for details.
OUT_CSV   = "/sharefs/lijl/xingjie/dataset/final_data_featurizable.csv"
CHUNK     = 250_000                              # Note: see the surrounding code for details.
# ----------------------------------------------------------------------

# Note: see the surrounding code for details.
bad_rows = np.fromfile(BAD_FILE, dtype=int, sep='\n')
bad_set  = set(bad_rows)              # Note: see the surrounding code for details.
print(f"Bad rows to drop : {len(bad_set):,}")

# Note: see the surrounding code for details.
tmpfile = Path(tempfile.mkstemp(suffix=".csv", dir=Path(OUT_CSV).parent)[1])
first   = True
abs_start = 0                         # Note: see the surrounding code for details.

TOTAL = sum(1 for _ in open(SRC_CSV, 'rb')) - 1   # Note: see the surrounding code for details.

with tqdm(total=TOTAL, unit="lines") as bar:
    for chunk in pd.read_csv(SRC_CSV, chunksize=CHUNK):
        n = len(chunk)
        abs_idx = np.arange(abs_start, abs_start + n)
        keep_mask = ~np.isin(abs_idx, bad_rows)   # Note: see the surrounding code for details.
        cleaned   = chunk.loc[keep_mask]

        cleaned.to_csv(tmpfile,
                       mode='w' if first else 'a',
                       header=first,
                       index=False)
        first = False
        abs_start += n
        bar.update(n)

# Note: see the surrounding code for details.
shutil.move(tmpfile, OUT_CSV)
print(f'Output file: {OUT_CSV}')
