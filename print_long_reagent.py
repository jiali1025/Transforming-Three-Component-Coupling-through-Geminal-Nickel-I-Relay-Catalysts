import pandas as pd

# Note: see the surrounding code for details.
csv_path = "/sharefs/lijl/xingjie/dataset/pistachio_filter.csv"

# Note: see the surrounding code for details.
df = pd.read_csv(csv_path, usecols=["id", "reagents", "reaction_smiles"])

# Note: see the surrounding code for details.
def count_components(cell: str) -> int:
    """English documentation for count_components."""
    return 0 if pd.isna(cell) else len([part for part in cell.split('.') if part])

df["n_reagents"] = df["reagents"].apply(count_components)

# Note: see the surrounding code for details.
df15 = df[df["n_reagents"] == 6]

print(f'Status: {len(df15):,}')

# Note: see the surrounding code for details.
for _, row in df15.head(10).iterrows():
    print("=" * 120)
    print(f'Status: {row.id} {row.n_reagents}')
    print("\nReagents:")
    print(row.reagents)              # Note: see the surrounding code for details.
    print("\nReaction SMILES:")
    print(row.reaction_smiles)       # Note: see the surrounding code for details.
    print()  # Note: see the surrounding code for details.