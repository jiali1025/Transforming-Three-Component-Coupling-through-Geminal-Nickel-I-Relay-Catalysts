from typing import List, Dict, Any
import numpy as np, torch
from chemprop.data import BatchMolGraph

def collate_graph_batch(samples: List[Dict[str, Any]]):
    # ------------ 1) CGR ------------
    cgr_bmg = BatchMolGraph([s["cgr"] for s in samples])

    # ------------ 2) Catalyst --------
    cat_lists   = [s["cat"] for s in samples]          # B x n_cat_i graphs.
    cat_flat    = [mg for sub in cat_lists for mg in sub]
    cat_bmg     = BatchMolGraph(cat_flat)              # Note: see the surrounding code for details.

    # Note: see the surrounding code for details.
    cat_graph2rxn = torch.repeat_interleave(
        torch.arange(len(samples)),                    # 0..B-1
        torch.tensor([len(lst) for lst in cat_lists])  # Note: see the surrounding code for details.
    )
    # Note: see the surrounding code for details.
    cat_node_batch = cat_graph2rxn[cat_bmg.batch]      # Note: see the surrounding code for details.

    # ------------ 3) Other reagent ---
    oth_lists   = [s["other"] for s in samples]
    oth_flat    = [mg for sub in oth_lists for mg in sub]
    oth_bmg     = BatchMolGraph(oth_flat)

    oth_graph2rxn = torch.repeat_interleave(
        torch.arange(len(samples)),
        torch.tensor([len(lst) for lst in oth_lists])
    )
    # Note: see the surrounding code for details.
    oth_graph_batch = oth_graph2rxn                    # length = #oth_graphs

    # ------------ 4) Targets ---------
    y = torch.tensor([s["y"] for s in samples], dtype=torch.float32).unsqueeze(1)

    return (cgr_bmg,                 # Note: see the surrounding code for details.
            cat_bmg, cat_node_batch, # node -> reaction index.
            oth_bmg, oth_graph_batch,# graph -> reaction index.
            y)