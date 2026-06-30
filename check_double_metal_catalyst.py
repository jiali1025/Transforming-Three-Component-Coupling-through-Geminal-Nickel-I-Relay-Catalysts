# count_bimetal_reactions.py
import pandas as pd
from rdkit import Chem

SRC = "/sharefs/lijl/xingjie/dataset/pistachio_filter.csv"   # Note: see the surrounding code for details.

# -------------------------------------------------------------
# Note: see the surrounding code for details.
# -------------------------------------------------------------
df = pd.read_csv(SRC, usecols=["reagents"])      # Note: see the surrounding code for details.

# -------------------------------------------------------------
# Note: see the surrounding code for details.
# -------------------------------------------------------------
METALS = {
     3, 11, 12, 13, 19, 20,            # Li Na Mg Al K Ca
    21,22,23,24,25,26,27,28,29,30,     # Note: see the surrounding code for details.
    39,40,41,42,43,44,45,46,47,48,     # Note: see the surrounding code for details.
    57,72,73,74,75,76,77,78,79,80      # Note: see the surrounding code for details.
}

def metals_in_smiles(smi: str) -> set[int]:
    """English documentation for metals_in_smiles."""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return set()
    return {a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() in METALS}

# -------------------------------------------------------------
# Note: see the surrounding code for details.
# -------------------------------------------------------------
n_same_mol  = 0   # Note: see the surrounding code for details.
n_two_metal = 0   # Note: see the surrounding code for details.

for reag_cell in df["reagents"].fillna(""):
    reagents = [r for r in reag_cell.split('.') if r]

    metals_per_mol = [metals_in_smiles(r) for r in reagents]

    # Note: see the surrounding code for details.
    if any(len(mset) >= 2 for mset in metals_per_mol):
        n_same_mol += 1

    # Note: see the surrounding code for details.
    if len(set().union(*metals_per_mol)) >= 2:
        n_two_metal += 1

# -------------------------------------------------------------
# Note: see the surrounding code for details.
# -------------------------------------------------------------
print('=== Summary ===')
print(f'Status: {n_same_mol:,}')
print(f'Status: {n_two_metal:,}')

# Section.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.

# Section.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.

# filter_pistachio
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.