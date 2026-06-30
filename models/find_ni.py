# find_ni_tm_quick.py
import pandas as pd
from rdkit import Chem

CSV = "/sharefs/lijl/xingjie/Ni_GAC_models/splits/test.csv"   # Note: see the surrounding code for details.
COL = "transition_metal_catalyst"

def has_ni(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return any(a.GetSymbol() == "Ni" for a in mol.GetAtoms())
    except Exception:
        return False

df = pd.read_csv(CSV)

# Note: see the surrounding code for details.
cand = df[COL].fillna("")
mask_regex = cand.str.contains(r"\[Ni", regex=True)

mask_rdkit = cand[mask_regex].apply(has_ni)
idx = cand[mask_regex][mask_rdkit].index

df_ni = df.loc[idx, ["reaction_smiles", COL, "other_reagent"]]

print(f'TM rows containing Ni: {len(df_ni)}')
print(df_ni.head(20).to_string(index=True))

# Note: see the surrounding code for details.
out = CSV.replace(".csv", "_with_Ni_TM.csv")
df_ni.to_csv(out, index=True)
print('Status message.', out)
