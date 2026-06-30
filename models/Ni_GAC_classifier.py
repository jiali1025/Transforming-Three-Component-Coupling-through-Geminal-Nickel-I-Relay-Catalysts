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

from models.Ni_GAC_model import GraphEncoder, MultiCrossGraphAttn
from combined_featurizers import CGR_FZR, MOL_FZR
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


def _build_metric_collection() -> nn.ModuleDict:
    """Create a fresh collection of binary classification metrics."""
    return nn.ModuleDict({
        "acc": BinaryAccuracy(),
        "ap": BinaryAveragePrecision(),
        "auroc": BinaryAUROC(),
        "f1": BinaryF1Score(),
    })


class YieldClassifier(pl.LightningModule):
    """
    LightningModule for classifying reactions into high/low yield.

    The architecture reuses the encoders and cross-graph attention from the
    regression model, but the prediction head is trained with a binary
    cross-entropy objective and reports common classification metrics.
    """

    def __init__(
        self,
        d_h: int = 1024,
        lr: float = 3e-4,
        n_attn_layers: int = 4,
        n_heads: int = 4,
        enc_depth: int = 3,
        threshold: float = 10.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters({
            "d_h": d_h,
            "lr": lr,
            "n_attn_layers": n_attn_layers,
            "n_heads": n_heads,
            "enc_depth": enc_depth,
            "threshold": threshold,
        })

        # ---------- encoders ----------
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

        # ---------- cross-attention ----------
        self.xattn = MultiCrossGraphAttn(
            d=d_h,
            n_heads=n_heads,
            n_layers=n_attn_layers,
            dropout=0.1,
            return_attn=False,
        )

        # ---------- classifier head ----------
        self.mlp_out = nn.Sequential(
            nn.Linear(d_h * 3, d_h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_metrics = _build_metric_collection()
        self.val_metrics = _build_metric_collection()
        self.test_metrics = _build_metric_collection()

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

        # 1) Encode reaction, catalyst, and other reagent graphs.
        H_cgr, _ = self.enc_cgr(cgr_bmg)
        H_cat, _ = self.enc_cat(cat_bmg)
        _, g_other = self.enc_other(oth_bmg)

        # 2) Cross attention between catalyst nodes and CGR nodes.
        z_cat, z_cgr = self.xattn(
            H_cat,
            cat_node_idx,
            H_cgr,
            cgr_bmg.batch,
        )

        # 3) Aggregate other reagent graphs on the reaction level.
        batch_size = z_cat.size(0)
        g_sum = torch.zeros(batch_size, g_other.size(1), device=self.device)
        g_sum.index_add_(0, oth_graph_idx, g_other)
        counts = torch.bincount(oth_graph_idx, minlength=batch_size).clamp(min=1).unsqueeze(1)
        z_other = g_sum / counts

        # 4) Concatenate and generate logits.
        logits = self.mlp_out(torch.cat([z_cgr, z_cat, z_other], dim=1)).squeeze(1)
        return logits

    # ---------------------------------------------------------------------
    # utilities
    # ---------------------------------------------------------------------
    def _shared_step(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self(batch)
        y_raw = batch[-1].squeeze(1)
        labels = (y_raw >= self.hparams.threshold)
        loss = self.loss_fn(logits, labels.to(logits.dtype))
        probs = torch.sigmoid(logits)
        return loss, probs, labels

    def _log_metrics(
        self,
        split: str,
        metrics: nn.ModuleDict,
        probs: Tensor,
        targets: Tensor,
        batch_size
    ) -> None:
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
                batch_size=batch_size,
            )

    # ---------------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        batch_size = targets.size(0)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self._log_metrics("train", self.train_metrics, probs, targets, batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        batch_size = targets.size(0)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self._log_metrics("val", self.val_metrics, probs, targets, batch_size)

    def test_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        batch_size = targets.size(0)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self._log_metrics("test", self.test_metrics, probs, targets, batch_size)

    def configure_optimizers(self):
        max_lr = self.hparams.lr
        total_steps = self.trainer.max_steps

        optimizer = AdamW(
            self.parameters(),
            lr=max_lr,
            weight_decay=1e-2,
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.10,
            div_factor=25,
            final_div_factor=1e3,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
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
