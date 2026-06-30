# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# ----------------------------------------------------------------------
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import lightning.pytorch as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryAUROC,
    BinaryF1Score,
)

from models.Ni_GAC_model import GraphEncoder   # Note: see the surrounding code for details.
from combined_featurizers import CGR_FZR, MOL_FZR
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


def _build_metric_collection() -> nn.ModuleDict:
    return nn.ModuleDict({
        "acc":   BinaryAccuracy(),
        "ap":    BinaryAveragePrecision(),
        "auroc": BinaryAUROC(),
        "f1":    BinaryF1Score(),
    })


# ---------------------------------------------------------------------------
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# Note: see the surrounding code for details.
# ---------------------------------------------------------------------------
class YieldClassifierNoAttn(pl.LightningModule):
    """Return a DataLoader for graph batches and targets."""

    def __init__(
        self,
        d_h: int = 1024,
        lr: float = 3e-4,
        enc_depth: int = 3,
        threshold: float = 10.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters({
            "d_h": d_h,
            "lr": lr,
            "enc_depth": enc_depth,
            "threshold": threshold,
        })

        # Note: see the surrounding code for details.
        self.enc_cgr = GraphEncoder(
            atom_fdim=CGR_FZR.atom_fdim,
            bond_fdim=CGR_FZR.bond_fdim,
            d_h=d_h,
            depth=enc_depth,
        )
        self.enc_cat = GraphEncoder(
            atom_fdim=MOL_FZR.atom_fdim,
            bond_fdim=MOL_FZR.bond_fdim,
            d_h=d_h,
            depth=enc_depth,
        )
        self.enc_other = GraphEncoder(
            atom_fdim=MOL_FZR.atom_fdim,
            bond_fdim=MOL_FZR.bond_fdim,
            d_h=d_h,
            depth=enc_depth,
        )

        # Note: see the surrounding code for details.

        # Note: see the surrounding code for details.
        self.mlp_out = nn.Sequential(
            nn.Linear(d_h * 3, d_h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, 1),
        )

        self.loss_fn       = nn.BCEWithLogitsLoss()
        self.train_metrics = _build_metric_collection()
        self.val_metrics   = _build_metric_collection()
        self.test_metrics  = _build_metric_collection()

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, batch) -> Tensor:
        (
            cgr_bmg,
            cat_bmg,
            cat_node_idx,
            oth_bmg,
            oth_graph_idx,
            _,
        ) = batch

        # Note: see the surrounding code for details.
        _, g_cgr   = self.enc_cgr(cgr_bmg)      # Note: see the surrounding code for details.
        H_cat, _   = self.enc_cat(cat_bmg)      # Note: see the surrounding code for details.
        _, g_other = self.enc_other(oth_bmg)    # subgraph x d_h.

        # Note: see the surrounding code for details.
        batch_size = g_cgr.size(0)
        z_cat = torch.zeros(batch_size, H_cat.size(1), device=self.device)
        z_cat.index_add_(0, cat_node_idx, H_cat)
        z_cat /= torch.bincount(cat_node_idx, minlength=batch_size).clamp(min=1).unsqueeze(1)

        # Note: see the surrounding code for details.
        g_sum  = torch.zeros(batch_size, g_other.size(1), device=self.device)
        g_sum.index_add_(0, oth_graph_idx, g_other)
        counts = torch.bincount(oth_graph_idx, minlength=batch_size).clamp(min=1).unsqueeze(1)
        z_other = g_sum / counts

        # Note: see the surrounding code for details.
        logits = self.mlp_out(
            torch.cat([g_cgr, z_cat, z_other], dim=1)
        ).squeeze(1)
        return logits

    # ---------------------------------------------------------------------
    # Note: see the surrounding code for details.
    # ---------------------------------------------------------------------
    def _shared_step(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self(batch)
        y_raw  = batch[-1].squeeze(1)
        labels = (y_raw >= self.hparams.threshold)
        loss   = self.loss_fn(logits, labels.to(logits.dtype))
        probs  = torch.sigmoid(logits)
        return loss, probs, labels

    def _log_metrics(
        self,
        split: str,
        metrics: nn.ModuleDict,
        probs: Tensor,
        targets: Tensor,
    ) -> None:
        bs = targets.size(0)
        for name, metric in metrics.items():
            metric(probs, targets)
            self.log(
                f"{split}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"acc", "auroc"},
                logger=True,
                sync_dist=True,
                batch_size=bs,
            )

    # ---------------------------------------------------------------------
    # Note: see the surrounding code for details.
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True,
            batch_size=targets.size(0),
        )
        self._log_metrics("train", self.train_metrics, probs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        self.log(
            "val_loss", loss,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True,
            batch_size=targets.size(0),
        )
        self._log_metrics("val", self.val_metrics, probs, targets)

    def test_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        self.log(
            "test_loss", loss,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True,
        )
        self._log_metrics("test", self.test_metrics, probs, targets)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-2,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.max_steps,
            pct_start=0.10,
            div_factor=25,
            final_div_factor=1e3,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def transfer_batch_to_device(self, batch, device, dtype=None):
        (
            cgr_bmg,
            cat_bmg,
            cat_node_idx,
            oth_bmg,
            oth_graph_idx,
            y,
        ) = batch
        cgr_bmg.to(device)
        cat_bmg.to(device)
        oth_bmg.to(device)
        return (
            cgr_bmg,
            cat_bmg,
            cat_node_idx.to(device),
            oth_bmg,
            oth_graph_idx.to(device),
            y.to(device),
        )
