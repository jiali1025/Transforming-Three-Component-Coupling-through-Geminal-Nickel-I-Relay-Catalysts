# merge_ionic_pairs_pipe.py
import pandas as pd
import numpy as np
from collections import OrderedDict
from rdkit import Chem

SRC = "/sharefs/lijl/xingjie/dataset/pistachio_merged_clean.csv"
DST = "/sharefs/lijl/xingjie/dataset/pistachio_ionic_merged.csv"

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
df = pd.read_csv(SRC)

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
def formal_charge(smi: str) -> int | None:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())

def merge_ionic_pairs(cell: str) -> str:
    """English documentation for merge_ionic_pairs."""
    if not isinstance(cell, str) or not cell.strip():
        return ""
    parts = [p.strip() for p in cell.split('.') if p.strip()]
    merged = []
    i = 0
    while i < len(parts):
        net = formal_charge(parts[i])
        if net is None:
            merged.append(parts[i])           # Note: see the surrounding code for details.
            i += 1
            continue

        j, net_j = i + 1, net
        while net_j != 0 and j < len(parts):
            q = formal_charge(parts[j])
            if q is None:
                break
            net_j += q
            j += 1

        if net_j == 0 and j > i + 1:          # Note: see the surrounding code for details.
            merged.append('.'.join(parts[i:j]))
            i = j
        else:                                 # Note: see the surrounding code for details.
            merged.append(parts[i])
            i += 1

    # Note: see the surrounding code for details.
    uniq = OrderedDict((z, None) for z in merged).keys()
    return '|'.join(uniq)                     # Note: see the surrounding code for details.

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
df["reagents"] = df["reagents"].apply(merge_ionic_pairs)
df["n_reagents"] = (df["reagents"]
                    .str.split('|')
                    .str.len()
                    .fillna(0)
                    .astype(np.uint8))

print('Status message.', df["n_reagents"].max())
print(df["n_reagents"].value_counts().sort_index())

###############################################################################
# Note: see the surrounding code for details.
###############################################################################
df.to_csv(DST, index=False)
print(f'Output file: {DST}')