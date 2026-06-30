import pandas as pd

# Note: see the surrounding code for details.
csv_path = "/sharefs/lijl/xingjie/dataset/pistachio_ionic_merged.csv"

# Note: see the surrounding code for details.
df = pd.read_csv(csv_path, usecols=["id", "reagents", "reaction_smiles"])

# Note: see the surrounding code for details.
mask_nacl = df["reagents"].fillna("").str.split('|').apply(lambda lst: "[Na+].[Cl-]" in lst)
df_nacl   = df[mask_nacl]

print(f'Records containing [Na+].[Cl-]: {len(df_nacl):,}')

# Note: see the surrounding code for details.
for _, row in df_nacl.sample(min(5, len(df_nacl))).iterrows():
    print("=" * 120)
    print(f"ID: {row.id}")
    print("\nReagents:")
    print(row.reagents)
    print("\nReaction SMILES:")
    print(row.reaction_smiles)          # Note: see the surrounding code for details.
    print()