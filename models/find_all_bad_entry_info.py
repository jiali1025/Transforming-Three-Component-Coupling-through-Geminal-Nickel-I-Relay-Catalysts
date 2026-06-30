# -*- coding: utf-8 -*-
"""Filter rows by the number of molecules in other_reagent."""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem, RDLogger, rdBase

# Note: see the surrounding code for details.
RDLogger.DisableLog('rdApp.*')
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data.datapoints import MoleculeDatapoint

# Note: see the surrounding code for details.
BAD_RAW = "bad_rows_raw.csv"       # Note: see the surrounding code for details.
OUT_TXT = "bad_molecules_set.txt"  # Note: see the surrounding code for details.
COLS    = ["transition_metal_catalyst", "other_reagent"]
CHUNK   = 50_000                   # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
mol_fzr       = SimpleMoleculeMolGraphFeaturizer()
bad_smiles_set = set()

def try_featurize(smi: str) -> bool:
    """English documentation for try_featurize."""
    try:
        dp = MoleculeDatapoint.from_smi(smi)
        _  = mol_fzr(dp.mol)        # Note: see the surrounding code for details.
        return True
    except Exception:
        return False

# Note: see the surrounding code for details.
for chunk in tqdm(pd.read_csv(BAD_RAW, usecols=COLS, chunksize=CHUNK),
                  desc="Scanning bad rows"):
    for col in COLS:
        for cell in chunk[col]:
            if not isinstance(cell, str) or not cell.strip():
                continue
            for smi in (s.strip() for s in cell.split('.') if s.strip()):
                if smi in bad_smiles_set:
                    continue               # Note: see the surrounding code for details.
                if not try_featurize(smi): # Note: see the surrounding code for details.
                    bad_smiles_set.add(smi)

# Note: see the surrounding code for details.
Path(OUT_TXT).write_text("\n".join(sorted(bad_smiles_set)))
print(f'Collected molecules requiring fixes: {len(bad_smiles_set):,} {OUT_TXT}')
