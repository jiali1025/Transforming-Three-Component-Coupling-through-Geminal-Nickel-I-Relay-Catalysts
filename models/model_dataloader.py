from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from new_dataset import RxnGraphDataset
from new_collate import collate_graph_batch


def build_loader(csv_path: str | Path,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 shuffle: bool = True):
    """Convert per-sample subgraph counts into prefix-sum offsets."""
    df = pd.read_csv(csv_path)

    ds = RxnGraphDataset(df)
    loader = DataLoader(
        ds,
        batch_size   = batch_size,
        shuffle      = shuffle,
        num_workers  = num_workers,
        pin_memory   = True,
        collate_fn   = collate_graph_batch,
    )
    return loader
