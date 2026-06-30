import pandas as pd
from rdkit import Chem
import random

# --------------------------
# Note: see the surrounding code for details.
# --------------------------
SRC = "/sharefs/lijl/xingjie/dataset/pistachio_filter.csv"  # Note: see the surrounding code for details.

CATALYTIC_METALS = {
    22, 23, 24, 25, 26, 27, 28, 29, 30,
    42, 44, 45, 46, 47, 48,
    72, 74, 76, 77, 78, 79,
    13, 31
}

# --------------------------
# Note: see the surrounding code for details.
# --------------------------
def is_likely_metal_catalyst(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return False
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    metal_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() in CATALYTIC_METALS]
    if 25 in atomic_nums:  # Note: see the surrounding code for details.
        if atomic_nums.count(8) >= 4:
            return False
    if not metal_atoms:
        return False
    ligand_like_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in {"P", "N", "O", "Cl", "Br", "F", "I"})
    has_ring = any(a.IsInRing() for a in mol.GetAtoms())
    return ligand_like_count >= 2 or has_ring

# --------------------------
# Note: see the surrounding code for details.
# --------------------------
df = pd.read_csv(SRC)
df["reagents"] = df["reagents"].fillna("")

# --------------------------
# Note: see the surrounding code for details.
# --------------------------
tmc_list = []
other_list = []

for reag_str in df["reagents"]:
    reagents = [r for r in reag_str.split('.') if r]
    tmc = []
    other = []
    for r in reagents:
        if is_likely_metal_catalyst(r):
            tmc.append(r)
        else:
            other.append(r)
    tmc_list.append('.'.join(tmc))
    other_list.append('.'.join(other))

df["transition_metal_catalyst"] = tmc_list
df["other_reagent"] = other_list

# --------------------------
# Note: see the surrounding code for details.
# --------------------------
all_tmc_mols = set()
for item in tmc_list:
    all_tmc_mols.update(r for r in item.split('.') if r)

sampled = random.sample(list(all_tmc_mols), min(20, len(all_tmc_mols)))
print('=== Summary ===')
for i, smi in enumerate(sampled, 1):
    print(f"{i:02d}: {smi}")

# --------------------------
# Note: see the surrounding code for details.
# --------------------------
output_df = df[["reaction_smiles", "transition_metal_catalyst", "other_reagent", "yield"]]
output_df.to_csv("/sharefs/lijl/xingjie/dataset/reagents_split_by_catalyst.csv", index=False)
print(" Saved full dataset: reagents_split_by_catalyst.csv")

# --------------------------
# Note: see the surrounding code for details.
# --------------------------
sample_1000 = output_df.sample(n=1000, random_state=42)
sample_1000.to_csv("/sharefs/lijl/xingjie/dataset/reagents_split_sample_1000.csv", index=False)
print(" Saved 1000-sample dataset: reagents_split_sample_1000.csv")
