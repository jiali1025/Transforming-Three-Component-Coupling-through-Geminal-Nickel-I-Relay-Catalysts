import pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')              # Note: see the surrounding code for details.
rdBase.DisableLog('rdApp.error')            # Note: see the surrounding code for details.
rdBase.DisableLog('rdApp.warning')          # Note: see the surrounding code for details.

from chemprop.featurizers import CondensedGraphOfReactionFeaturizer, SimpleMoleculeMolGraphFeaturizer
from chemprop.data.datapoints import ReactionDatapoint, MoleculeDatapoint

# Configuration.
CSV_PATH   = "/sharefs/lijl/xingjie/dataset/final_data_featurizable_max.csv"
OUT_BADIDX = Path("bad_rows_final_check.txt")
SAMPLE_N   = 100
CHUNK      = 200_000                   # Note: see the surrounding code for details.
MINITER    = 1_000                     # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
with open(CSV_PATH, 'rb') as f:
    N_TOTAL = sum(1 for _ in f) - 1
print(f"Total lines (excluding header): {N_TOTAL:,}")

# Note: see the surrounding code for details.
cgr_fzr = CondensedGraphOfReactionFeaturizer()
mol_fzr = SimpleMoleculeMolGraphFeaturizer()

# Note: see the surrounding code for details.
reservoir, bad_rows, bad_cnt = [], [], 0
rng = np.random.default_rng()

def add_sample(rec):
    global bad_cnt, reservoir
    bad_cnt += 1
    if len(reservoir) < SAMPLE_N:
        reservoir.append(rec)
    else:
        j = rng.integers(bad_cnt)
        if j < SAMPLE_N:
            reservoir[j] = rec

# Note: see the surrounding code for details.
def test_cgr(rxn_smi, row):
    try:
        dp = ReactionDatapoint.from_smi(rxn_smi)
        _  = cgr_fzr((dp.rct, dp.pdt))
        return False
    except Exception as e:
        add_sample((row, "CGR", rxn_smi, str(e)))
        return True

def test_smiles_cell(cell, row, role):
    if not isinstance(cell, str) or not cell.strip():
        return False
    for smi in cell.split('.'):
        s = smi.strip()
        if not s:
            continue
        try:
            dp = MoleculeDatapoint.from_smi(s)
            _  = mol_fzr(dp.mol)
        except Exception as e:
            add_sample((row, role, s, str(e)))
            return True
    return False

# Note: see the surrounding code for details.
usecols = ["reaction_smiles", "transition_metal_catalyst", "other_reagent"]
row_global = 0

with tqdm(total=N_TOTAL, unit="lines", miniters=MINITER) as pbar:
    for chunk in pd.read_csv(CSV_PATH, usecols=usecols, chunksize=CHUNK):
        for _, r in chunk.iterrows():
            bad = (
                test_cgr(r.reaction_smiles, row_global)
                or test_smiles_cell(r.transition_metal_catalyst, row_global, "Cat")
                or test_smiles_cell(r.other_reagent,             row_global, "Other")
            )
            if bad:
                bad_rows.append(str(row_global))
            row_global += 1
            pbar.update(1)                 # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
OUT_BADIDX.write_text("\n".join(bad_rows))
print(f"\n=== Done ===")
print(f"Bad rows : {len(bad_rows):,}")
print(f"Indices  saved to: {OUT_BADIDX.resolve()}")

print(f"\n=== Random {len(reservoir)} bad samples ===")
for i, (idx, col, badstr, msg) in enumerate(reservoir, 1):
    print(f"{i:02d}) row {idx:,} | {col}\n    {badstr}\n    -> {msg}\n")
