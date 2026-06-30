import pandas as pd
from rdkit import Chem

csv_path = "/sharefs/lijl/xingjie/dataset/pistachio_ionic_merged.csv"
df = pd.read_csv(csv_path, usecols=["id", "reagents"])

# Note: see the surrounding code for details.
METALS        = {3, 11, 12, 13, 19, 20, 30, 31}      # Li Na Mg Al K Ca Zn Ga.
COORD_ATOMS   = {7, 8, 9, 15, 16, 17}                # N O F P S Cl
# ----------------------

flag_rows_charge, flag_rows_nodot, detail_nodot = [], [], []

def formal_charge(smi: str) -> int | None:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())

for row in df.itertuples():
    reag_cell = row.reagents
    if not isinstance(reag_cell, str) or not reag_cell.strip():   # Safeguard.
        continue

    total_charge = 0
    for smi in reag_cell.split('|'):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            continue

        # Note: see the surrounding code for details.
        total_charge += sum(a.GetFormalCharge() for a in mol.GetAtoms())

        # Note: see the surrounding code for details.
        if mol.GetNumAtoms() > 1 and len(Chem.GetMolFrags(mol)) == 1:
            nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
            if nums & METALS and nums & COORD_ATOMS and Chem.GetFormalCharge(mol) == 0:
                flag_rows_nodot.append(row.id)
                detail_nodot.append((row.id, smi))

    if total_charge != 0:
        flag_rows_charge.append(row.id)

print(f'Charge-imbalanced reactions: {len(flag_rows_charge):,}')
print(f'Possible salts without dot separators: {len(set(flag_rows_nodot)):,}')
print(f'Suspicious reagent count: {len(detail_nodot):,}')

# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
