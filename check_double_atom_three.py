#!/usr/bin/env python3
# three_substrate_double-metal.py
# -------------------------------------------------------------
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# -------------------------------------------------------------

import pandas as pd
from rdkit import Chem
from collections import Counter

# Note: see the surrounding code for details.
CSV_PATH = "/sharefs/lijl/xingjie/dataset/pistachio_filter.csv"  # Note: see the surrounding code for details.
SHOW_N   = 20           # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
TM = (
    set(range(21, 31))   |   # Sc to Zn.
    set(range(39, 49))   |   # Y to Cd.
    {57}                 |   # Note: see the surrounding code for details.
    set(range(72, 81))       # Hf to Au.
)

# Note: see the surrounding code for details.
def n_reactants(reactants: str) -> int:
    """English documentation for n_reactants."""
    return len([frag for frag in reactants.split('.') if frag])

def metal_atom_count(smi: str) -> int:
    """English documentation for metal_atom_count."""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in TM) if mol else 0

def max_metal_atoms_in_reagents(reagent_str: str) -> int:
    """Return the maximum metal-atom count among reagent molecules split by '|'."""
    if not isinstance(reagent_str, str):
        return 0
    return max(
        (metal_atom_count(smi) for smi in reagent_str.split('.') if smi),
        default=0
    )

# Note: see the surrounding code for details.
print("Loading CSV ...")
df = pd.read_csv(
    CSV_PATH,
    usecols=["id", "reactants", "reagents", "reaction_smiles"],
    low_memory=False
)

# Note: see the surrounding code for details.
df_three = df[df["reactants"].apply(n_reactants)>0].copy()
print(f'Three-reactant reaction count: {len(df_three):,}')

# Note: see the surrounding code for details.
df_three["max_metal_atoms"] = df_three["reagents"].apply(max_metal_atoms_in_reagents)

# Note: see the surrounding code for details.
dist = Counter(df_three["max_metal_atoms"])
print('Status message.')
total = len(df_three)
for k in sorted(dist):
    tag = {0:'0 metals', 1:'1 metal', 2:'>=2 metals'}.get(k, '>=2 metals')
    print(f'Distribution row: {tag:<6} {dist[k]:,} {dist[k]/total:.2%}')

# Note: see the surrounding code for details.
hits = df_three[df_three["max_metal_atoms"] >= 2]
if SHOW_N and not hits.empty:
    print(f'Random sample: {min(SHOW_N, len(hits))}')
    for _, row in hits.sample(n=min(SHOW_N, len(hits)), random_state=42).iterrows():
        print("=" * 120)
        print(f"ID: {row.id}")
        print("\nReagents:\n", row.reagents)
        print("\nReaction SMILES:\n", row.reaction_smiles, "\n")
else:
    print('Status message.')
