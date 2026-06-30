# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# ----------------------------------------------------------------------
from __future__ import annotations
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import lightning.pytorch as pl

from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn import MeanAggregation
from chemprop.data import BatchMolGraph
from combined_featurizers import CGR_FZR, MOL_FZR
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class GraphEncoder(nn.Module):
    """
    BatchMolGraph -> ( node_emb , graph_emb )
    node_emb : V x d_h
    graph_emb: B x d_h  (mean-pool over nodes)
    """
    def __init__(self, atom_fdim: int, bond_fdim: int, d_h: int = 300, depth: int = 3):
        super().__init__()
        self.mp = BondMessagePassing(
            d_v=atom_fdim,
            d_e=bond_fdim,
            d_h=d_h,
            depth=depth,
        )
        self.readout = MeanAggregation()

    @property
    def d_out(self) -> int:
        return self.mp.output_dim

    def forward(self, bmg: BatchMolGraph) -> Tuple[Tensor, Tensor]:
        H_v = self.mp(bmg)                   # V x d_h.
        g   = self.readout(H_v, bmg.batch)   # B x d_h.
        return H_v, g


# ---------------------------------------------------------------------------
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# ---------------------------------------------------------------------------
class YieldPredictorNoAttn(pl.LightningModule):
    """Return a DataLoader for graph batches and targets."""

    def __init__(
        self,
        d_h: int = 300,
        lr: float = 1e-4,
        *,
        attn_dropout: float = 0.1,   # Note: see the surrounding code for details.
    ):
        super().__init__()
        self.save_hyperparameters()

        # Note: see the surrounding code for details.
        self.enc_cgr   = GraphEncoder(CGR_FZR.atom_fdim,  CGR_FZR.bond_fdim,  d_h)
        self.enc_cat   = GraphEncoder(MOL_FZR.atom_fdim,  MOL_FZR.bond_fdim,  d_h)
        self.enc_other = GraphEncoder(MOL_FZR.atom_fdim,  MOL_FZR.bond_fdim,  d_h)

        # Note: see the surrounding code for details.
        self.cat_other_proj = nn.Linear(d_h * 2, d_h)

        # Note: see the surrounding code for details.
        self.mlp_out = nn.Sequential(
            nn.Linear(d_h * 3, d_h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, 1),
        )

        self.lr      = lr
        self.loss_fn = nn.MSELoss()

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, batch) -> Tensor:
        (cgr_bmg, cat_bmg, cat_node_idx,
         oth_bmg, oth_graph_idx, _) = batch

        # Note: see the surrounding code for details.
        _, g_cgr     = self.enc_cgr(cgr_bmg)     # Note: see the surrounding code for details.
        H_cat, _     = self.enc_cat(cat_bmg)     # Note: see the surrounding code for details.
        _, g_other   = self.enc_other(oth_bmg)   # subgraph x d_h.

        # Note: see the surrounding code for details.
        B = cgr_bmg.batch.max().item() + 1
        g_sum  = torch.zeros(B, g_other.size(1), device=g_other.device)
        g_sum.index_add_(0, oth_graph_idx, g_other)
        counts = torch.bincount(oth_graph_idx, minlength=B).clamp(min=1).unsqueeze(1)
        z_oth  = g_sum / counts                  # B x d_h.

        # Note: see the surrounding code for details.
        g_other_nodes = z_oth[cat_node_idx]                           # V x d_h.
        H_cat = self.cat_other_proj(
            torch.cat([H_cat, g_other_nodes], dim=1)
        )                                                              # V x d_h.

        # Note: see the surrounding code for details.
        z_cat = torch.zeros(B, H_cat.size(1), device=H_cat.device)
        z_cat.index_add_(0, cat_node_idx, H_cat)
        z_cat /= torch.bincount(cat_node_idx, minlength=B).clamp(min=1).unsqueeze(1)
        # B x d_h.

        # Note: see the surrounding code for details.
        y_hat = self.mlp_out(
            torch.cat([g_cgr, z_cat, z_oth], dim=1)
        ).squeeze(1)
        return y_hat

    # ---------------- Lightning hooks ----------------
    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        y     = batch[-1].squeeze(1)
        loss  = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True,
                 batch_size=y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        y     = batch[-1].squeeze(1)
        loss  = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=y.size(0))

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        sched = OneCycleLR(
            opt,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.max_steps,
            pct_start=0.10,
            div_factor=25,
            final_div_factor=1e3,
            anneal_strategy="cos",
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def transfer_batch_to_device(self, batch, device, dtype=None):
        (cgr_bmg, cat_bmg, cat_node_idx,
         oth_bmg, oth_graph_idx, y) = batch
        cgr_bmg.to(device)
        cat_bmg.to(device)
        oth_bmg.to(device)
        return (cgr_bmg,
                cat_bmg,
                cat_node_idx.to(device),
                oth_bmg,
                oth_graph_idx.to(device),
                y.to(device))
