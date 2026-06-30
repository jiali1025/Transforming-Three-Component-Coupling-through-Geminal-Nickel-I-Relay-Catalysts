import pandas as pd
from pathlib import Path

CSV_PATH   = "/sharefs/lijl/xingjie/dataset/reagents_split_by_catalyst.csv"
IDX_PATH   = "bad_row_indices.txt"
BAD_RAW    = "bad_rows_raw.csv"

bad_idx = set(map(int, Path(IDX_PATH).read_text().splitlines()))
usecols = ["reaction_smiles","transition_metal_catalyst",
           "other_reagent","yield"]

rows = []
for chunk in pd.read_csv(CSV_PATH, usecols=usecols, chunksize=200_000):
    mask = chunk.index.isin(bad_idx)
    rows.append(chunk[mask])

bad_df = pd.concat(rows, ignore_index=True)
bad_df.to_csv(BAD_RAW, index=False)
print(f'Extracted bad samples: {len(bad_df):,} {BAD_RAW}')
