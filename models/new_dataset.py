# dataset.py
from torch.utils.data import Dataset
import pandas as pd
from combined_featurizers import smi_to_molgraph, rxn_smi_to_cgr, EMPTY_GRAPH

class RxnGraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Note: see the surrounding code for details.
        cgr = rxn_smi_to_cgr(row.reaction_smiles)

        # Note: see the surrounding code for details.
        def split_and_conv(cell: str | float):
            if not isinstance(cell, str) or not cell.strip():
                return [EMPTY_GRAPH]
            return [smi_to_molgraph(s) for s in cell.split('.') if s]

        cat   = split_and_conv(row.transition_metal_catalyst)
        other = split_and_conv(row.other_reagent)

        y = float(row["yield"])

        return {"cgr": cgr, "cat": cat, "other": other, "y": y}
