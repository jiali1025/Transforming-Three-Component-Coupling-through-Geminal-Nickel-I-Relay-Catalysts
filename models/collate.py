# collate.py
from typing import List, Dict, Any
import numpy as np
import torch
from chemprop.data import BatchMolGraph

def _build_offsets(lengths: list[int]) -> torch.Tensor:
    """Convert per-sample subgraph counts into prefix-sum offsets."""
    return torch.tensor(np.concatenate([[0], np.cumsum(lengths)]), dtype=torch.long)

def collate_graph_batch(samples: List[Dict[str, Any]]):


    # Note: see the surrounding code for details.
    cgr_bmg = BatchMolGraph([s["cgr"] for s in samples])

    # Note: see the surrounding code for details.
    cat_lists  = [s["cat"]   for s in samples]
    cat_lens   = [len(x)     for x in cat_lists]            # Note: see the surrounding code for details.
    cat_flat   = [mg for sub in cat_lists for mg in sub]
    cat_bmg    = BatchMolGraph(cat_flat)
    cat_offsets = _build_offsets(cat_lens)                   # Note: see the surrounding code for details.

    # Note: see the surrounding code for details.
    oth_lists  = [s["other"] for s in samples]
    oth_lens   = [len(x)     for x in oth_lists]
    oth_flat   = [mg for sub in oth_lists for mg in sub]
    oth_bmg    = BatchMolGraph(oth_flat)
    oth_offsets = _build_offsets(oth_lens)

    # 4) Targets
    y = torch.tensor([s["y"] for s in samples], dtype=torch.float32).unsqueeze(1)

    return cgr_bmg, cat_bmg, cat_offsets, oth_bmg, oth_offsets, y
