import torch
from pathlib import Path
from model_dataloader import build_loader   # Note: see the surrounding code for details.
from combined_featurizers import CGR_FZR, MOL_FZR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV = PROJECT_ROOT / "splits" / "train.csv"
loader = build_loader(CSV, batch_size=8, num_workers=0, shuffle=True)
batch = next(iter(loader))

(cgr_bmg,
 cat_bmg, cat_node2rxn,
 oth_bmg, oth_graph2rxn,
 y) = batch

print("=== Batch shapes ===")
print(f"CGR atoms   : {cgr_bmg.V.shape} (atom_fdim={CGR_FZR.atom_fdim})")
print(f"Catalyst atoms : {cat_bmg.V.shape} (atom_fdim={MOL_FZR.atom_fdim})")
print(f"Other graphs  : {len(torch.unique(oth_graph2rxn))}")
print(f"Target y      : {y.shape}")

# Note: see the surrounding code for details.
rxn0 = 0
n_cat0 = torch.unique(cat_bmg.batch[cat_node2rxn == rxn0]).numel()
print(f"\nReaction-0  catalyst graphs : {n_cat0}")

# ====== move to CUDA ======
device = "cuda" if torch.cuda.is_available() else "cpu"
for obj in (cgr_bmg, cat_bmg, oth_bmg):
    obj.to(device)
y = y.to(device)
print("\n tensors on", device)
