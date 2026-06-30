import pandas as pd
from rdkit import Chem
from collections import Counter

# Note: see the surrounding code for details.
CSV_PATH = "/sharefs/lijl/xingjie/dataset/pistachio_filter.csv"   # Note: see the surrounding code for details.
N_SHOW   = 20      # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
METALS = {
     3,11,12,13,19,20,                 # Note: see the surrounding code for details.
    21,22,23,24,25,26,27,28,29,30,     # 3d
    39,40,41,42,43,44,45,46,47,48,     # 4d
    57,72,73,74,75,76,77,78,79,80      # Note: see the surrounding code for details.
}

# Note: see the surrounding code for details.
def metals_in_smiles(smi: str) -> set[int]:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    return {a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() in METALS} if mol else set()

def n_reactants(rec: str) -> int:
    return len([frag for frag in rec.split('.') if frag])

def max_metal_kinds_in_reagents(reag: str) -> int:
    if not isinstance(reag, str):
        return 0
    return max(
        (len(metals_in_smiles(smi)) for smi in reag.split('.') if smi),
        default=0
    )

# Note: see the surrounding code for details.
print("Loading CSV ...")
df = pd.read_csv(CSV_PATH, usecols=["id", "reactants", "reagents", "reaction_smiles"])

print("Filtering three-substrate reactions ...")
mask_three = df["reactants"].apply(n_reactants) == 3
df3 = df.loc[mask_three].copy()
print(f'Three-reactant reaction count: {len(df3):,}')

print("Calculating metal stats ...")
df3["max_metal_in_one_reagent"] = df3["reagents"].apply(max_metal_kinds_in_reagents)

# Note: see the surrounding code for details.
dist = Counter(df3["max_metal_in_one_reagent"])
print('Status message.')
labels = {0:'no metal',1:'single metal',2:'two or more metals'}
total  = len(df3)
for k in sorted(dist):
    tag = labels.get(k, f'metals: {k}')
    print(f'Distribution row: {tag:<6} {dist[k]:,} {dist[k]/total:.2%}')

# Note: see the surrounding code for details.
df_bi = df3[df3["max_metal_in_one_reagent"] >= 2]
print(f'Random sample: {min(N_SHOW,len(df_bi))}')

for _, row in df_bi.sample(min(N_SHOW, len(df_bi)), random_state=42).iterrows():
    print("=" * 120)
    print(f"ID: {row.id}")
    print("\nReagents:")
    print(row.reagents)
    print("\nReaction SMILES:")
    print(row.reaction_smiles)
    print()   # Note: see the surrounding code for details.