# Note: see the surrounding code for details.
# ----------------------------------------------------------------------
from __future__ import annotations
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import lightning.pytorch as pl

from chemprop.nn.message_passing import BondMessagePassing     # Note: see the surrounding code for details.
from chemprop.nn import MeanAggregation
from chemprop.data import BatchMolGraph
from combined_featurizers import CGR_FZR, MOL_FZR
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class GraphEncoder(nn.Module):
    """English documentation for GraphEncoder."""
    def __init__(self, atom_fdim: int, bond_fdim: int,  d_h: int = 300, depth: int = 3):
        super().__init__()
        self.mp = BondMessagePassing(
            d_v=atom_fdim,           # Note: see the surrounding code for details.
            d_e=bond_fdim,
            d_h=d_h,
            depth=depth,
        )
        self.readout = MeanAggregation()

    @property
    def d_out(self) -> int:
        return self.mp.output_dim

    def forward(self, bmg: BatchMolGraph) -> Tuple[Tensor, Tensor]:
        H_v = self.mp(bmg)                       # Note: see the surrounding code for details.
        g = self.readout(H_v, bmg.batch)         # Note: see the surrounding code for details.
        return H_v, g

# ----------------------------------------------------------------------
# Note: see the surrounding code for details.
# ----------------------------------------------------------------------
class _AttnLayer(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.h = n_heads
        self.scale = (d // n_heads) ** 0.5
        self.do = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(d)
        self.norm_b = nn.LayerNorm(d)

    def _split(self, x):                 # (N, d) -> (h, N, d / h).
        h = self.h
        return x.reshape(x.size(0), h, -1).transpose(0, 1)

    def forward(self, H_a, H_b, mask):   # mask:  bool(Na,Nb)
        Qa, Kb, Vb = (self._split(p(H)) for p, H in
                      [(self.q_proj, H_a), (self.k_proj, H_b), (self.v_proj, H_b)])
        Qb, Ka, Va = (self._split(p(H)) for p, H in
                      [(self.q_proj, H_b), (self.k_proj, H_a), (self.v_proj, H_a)])

        attn_ab = torch.einsum("hqd,hkd->hqk", Qa, Kb) / self.scale
        attn_ba = torch.einsum("hqd,hkd->hqk", Qb, Ka) / self.scale
        attn_ab.masked_fill_(~mask, -1e4)
        attn_ba.masked_fill_(~mask.t(), -1e4)

        w_ab = attn_ab.softmax(-1)                     # heads x Na x Nb.
        w_ba = attn_ba.softmax(-1)                     # heads x Nb x Na.

        # output: same shape as input nodes
        Za = torch.einsum("hqk,hkd->hqd", w_ab, Vb).transpose(0,1).reshape_as(H_a)
        Zb = torch.einsum("hqk,hkd->hqd", w_ba, Va).transpose(0,1).reshape_as(H_b)

        H_a = self.norm_a(H_a + self.do(Za))
        H_b = self.norm_b(H_b + self.do(Zb))
        return H_a, H_b, w_ab, w_ba                    # Note: see the surrounding code for details.

class MultiCrossGraphAttn(nn.Module):
    """English documentation for MultiCrossGraphAttn."""
    def __init__(self, d, n_heads=4, n_layers=2, dropout=0.1,
                 return_attn: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(
            [_AttnLayer(d, n_heads, dropout) for _ in range(n_layers)]
        )
        self.return_attn = return_attn
        self.last_attn: list[dict[str, Tensor]] | dict[str, Tensor] | None = None

    def forward(self, H_a, batch_a, H_b, batch_b):
        # Note: see the surrounding code for details.
        mask = (batch_a.unsqueeze(1) == batch_b.unsqueeze(0))   # (Na,Nb)

        attn_collector = [] if self.return_attn else None

        for layer in self.layers:
            H_a, H_b, w_ab, w_ba = layer(H_a, H_b, mask)

            if self.return_attn:
                attn_collector.append({"ab": w_ab.detach(),
                                       "ba": w_ba.detach()})

        # ------ pooling ------
        z_a = torch.zeros(batch_a.max()+1, H_a.size(1), device=H_a.device)
        z_a.index_add_(0, batch_a, H_a)
        z_a /= torch.bincount(batch_a, minlength=len(z_a)).unsqueeze(1)

        z_b = torch.zeros(batch_b.max()+1, H_b.size(1), device=H_b.device)
        z_b.index_add_(0, batch_b, H_b)
        z_b /= torch.bincount(batch_b, minlength=len(z_b)).unsqueeze(1)

        # Note: see the surrounding code for details.
        if self.return_attn:
            self.last_attn = attn_collector
        else:
            self.last_attn = {"ab": w_ab.detach(), "ba": w_ba.detach()}

        return z_a, z_b

# ---------------------------------------------------------------------------
# Note: see the surrounding code for details.
# ---------------------------------------------------------------------------
class YieldPredictor(pl.LightningModule):
    """Return a DataLoader for graph batches and targets."""

    def __init__(
        self,
        d_h: int = 300,
        lr: float = 1e-4,
        *,                       # Note: see the surrounding code for details.
        n_attn_layers: int = 3,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        return_attn: bool = False,
    ):
        super().__init__()
        # Note: see the surrounding code for details.
        self.save_hyperparameters()

        # ---------- graph encoders ----------
        self.enc_cgr   = GraphEncoder(CGR_FZR.atom_fdim,  CGR_FZR.bond_fdim,  d_h)
        self.enc_cat   = GraphEncoder(MOL_FZR.atom_fdim,  MOL_FZR.bond_fdim,  d_h)
        self.enc_other = GraphEncoder(MOL_FZR.atom_fdim,  MOL_FZR.bond_fdim,  d_h)

        # ---------- multi-layer cross-attention ----------
        self.xattn = MultiCrossGraphAttn(
            d=d_h,
            n_heads=n_heads,
            n_layers=n_attn_layers,
            dropout=attn_dropout,
            return_attn=return_attn,
        )
        self.cat_other_proj = nn.Linear(d_h * 2, d_h)  # Note: see the surrounding code for details.

        # ---------- head ----------
        self.mlp_out = nn.Sequential(
            nn.Linear(d_h * 3, d_h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, 1),
        )

        self.lr       = lr
        self.loss_fn  = nn.MSELoss()

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, batch) -> Tensor:
        (cgr_bmg, cat_bmg, cat_node_idx,
         oth_bmg, oth_graph_idx, _) = batch

        # Note: see the surrounding code for details.
        H_cgr, _   = self.enc_cgr(cgr_bmg)      # node-level
        H_cat, _   = self.enc_cat(cat_bmg)
        _, g_other = self.enc_other(oth_bmg)    # graph-level only

        # Note: see the surrounding code for details.

        # reaction-level other pooling (shared later by fusion + regression head)
        B = cgr_bmg.batch.max().item() + 1
        g_sum = torch.zeros(B, g_other.size(1), device=g_other.device)
        g_sum.index_add_(0, oth_graph_idx, g_other)
        counts = torch.bincount(oth_graph_idx, minlength=B).clamp(min=1).unsqueeze(1)
        z_oth = g_sum / counts                   # B x d_h

        # Note: see the surrounding code for details.
        # Catalyst nodes absorb reaction-level other context prior to attention
        g_other_nodes = z_oth[cat_node_idx] # Broadcast batch features to node features.
        H_cat = torch.cat([H_cat, g_other_nodes], dim=1) # Concatenate node features.
        H_cat = self.cat_other_proj(H_cat)  # Note: see the surrounding code for details.

        # Note: see the surrounding code for details.
        z_cat, z_cgr = self.xattn(
            H_cat, cat_node_idx,
            H_cgr, cgr_bmg.batch
        )                                        # B x d_h.

        # 5) concat & regress
        y_hat = self.mlp_out(torch.cat([z_cgr, z_cat, z_oth], dim=1)).squeeze(1)
        y_hat = 100 * torch.sigmoid(y_hat) # Note: see the surrounding code for details.
        return y_hat

    # ---------------- Lightning hooks ----------------
    def training_step(self, batch, batch_idx):  # Note: see the surrounding code for details.
        y_hat = self(batch)
        y = batch[-1].squeeze(1)
        loss = self.loss_fn(y_hat, y)

        # Note: see the surrounding code for details.
        # Note: see the surrounding code for details.
        # Note: see the surrounding code for details.
        self.log("train_loss",
                 loss,
                 on_step=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch[-1].squeeze(1)
        loss = self.loss_fn(y_hat, y)

        # Note: see the surrounding code for details.
        self.log("val_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def configure_optimizers(self):
        """English documentation for configure_optimizers."""
        max_lr = self.hparams.lr  # Note: see the surrounding code for details.
        total_steps = self.trainer.max_steps  # Note: see the surrounding code for details.

        # -------- Optimizer --------
        opt = AdamW(self.parameters(),
                    lr=max_lr,  # Note: see the surrounding code for details.
                    weight_decay=1e-2)

        # -------- Scheduler: One-Cycle --------
        sched = OneCycleLR(
            opt,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.10,  # Note: see the surrounding code for details.
            div_factor=25,  # Note: see the surrounding code for details.
            final_div_factor=1e3,  # Note: see the surrounding code for details.
            anneal_strategy="cos",
        )

        # Note: see the surrounding code for details.
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
            },
        }

    # ------------ utilities -------------
    def get_last_attention(self):
        """English documentation for get_last_attention."""
        return self.xattn.last_attn

    def transfer_batch_to_device(self, batch, device, dtype=None):
        """English documentation for transfer_batch_to_device."""
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
