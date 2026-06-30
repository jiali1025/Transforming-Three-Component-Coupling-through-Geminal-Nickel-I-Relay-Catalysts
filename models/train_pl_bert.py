# -*- coding: utf-8 -*-
import os
import math
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------- Logger -----------------------------
logger = logging.getLogger("bert")

@rank_zero_only
def setup_loggers(outdir: Path, log_level: str = 'INFO'):
    """English documentation for setup_loggers."""
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / 'training.log'

    logger.handlers.clear()
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Note: see the surrounding code for details.
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Note: see the surrounding code for details.
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f'Log file: {log_path}')

# ----------------------------- IO -----------------------------
def load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file: {path}")

def prepare_dataframe(df: pd.DataFrame, text_col: str, label_col: str, ctx: str) -> pd.DataFrame:
    need = {text_col, label_col}
    if not need.issubset(df.columns):
        raise ValueError(f"{ctx}: need columns {need}, got {list(df.columns)}")
    df = df[[text_col, label_col]].copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df.dropna(subset=[text_col, label_col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError(f"{ctx}: empty after cleaning")
    return df

# ------------------------- Dataset/Collate -------------------------
class SmilesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str,
                 tokenizer: AutoTokenizer, max_len: int, task: str):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.label_col = label_col
        self.tok = tokenizer
        self.max_len = max_len
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        text = str(r[self.text_col])
        enc = self.tok(text, truncation=True, max_length=self.max_len, padding=False, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        y = float(r[self.label_col])
        if self.task == "classification":
            item["labels"] = torch.tensor(int(y), dtype=torch.long)  # Note: see the surrounding code for details.
        else:
            item["labels"] = torch.tensor([y], dtype=torch.float)    # Note: see the surrounding code for details.
        return item

def dynamic_collate(batch: list) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        if k == "labels":
            # Note: see the surrounding code for details.
            if batch[0]["labels"].dim() == 0:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                out[k] = torch.cat([b[k] for b in batch], dim=0)
        else:
            tensors = [b[k] for b in batch]
            maxlen = max(t.size(-1) for t in tensors)
            padded = [F.pad(t, (0, maxlen - t.size(-1)), value=0) for t in tensors]
            out[k] = torch.stack(padded, dim=0)
    return out

# ----------------------- LightningModule -----------------------
try:
    import torchmetrics as tm
    TM_AVAILABLE = True
except Exception:
    TM_AVAILABLE = False

class YieldBERTModule(pl.LightningModule):
    def __init__(self,
                 model_name_or_path: str,
                 task: str = "regression",
                 lr: float = 1e-4,
                 dropout: float = 0.1,
                 from_scratch: bool = False,
                 offline: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.lr = lr
        self.offline = offline

        if from_scratch:
            cfg = AutoConfig.from_pretrained(
                model_name_or_path,
                local_files_only=self.offline
            )
            if task == "regression":
                cfg.num_labels = 1
                cfg.problem_type = "regression"
            else:
                cfg.num_labels = 2
                cfg.problem_type = "single_label_classification"
            if hasattr(cfg, "hidden_dropout_prob"):
                cfg.hidden_dropout_prob = dropout
            if hasattr(cfg, "attention_probs_dropout_prob"):
                cfg.attention_probs_dropout_prob = dropout
            self.model = AutoModelForSequenceClassification.from_config(cfg)
        else:
            cfg = AutoConfig.from_pretrained(
                model_name_or_path,
                local_files_only=self.offline
            )
            if task == "regression":
                cfg.num_labels = 1
                cfg.problem_type = "regression"
            else:
                cfg.num_labels = 2
                cfg.problem_type = "single_label_classification"
            if hasattr(cfg, "hidden_dropout_prob"):
                cfg.hidden_dropout_prob = dropout
            if hasattr(cfg, "attention_probs_dropout_prob"):
                cfg.attention_probs_dropout_prob = dropout
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, config=cfg, local_files_only=self.offline
            )

        # Metrics
        if TM_AVAILABLE:
            if task == "regression":
                self.val_mae = tm.MeanAbsoluteError()
                self.val_rmse = tm.MeanSquaredError(squared=False)
                self.test_mae = tm.MeanAbsoluteError()
                self.test_rmse = tm.MeanSquaredError(squared=False)
            else:
                self.val_acc = tm.Accuracy(task="multiclass", num_classes=2)
                self.val_f1 = tm.F1Score(task="multiclass", num_classes=2, average="macro")
                self.val_auroc = tm.AUROC(task="binary")
                self.val_ap = tm.AveragePrecision(task="binary")
                self.test_acc = tm.Accuracy(task="multiclass", num_classes=2)
                self.test_f1 = tm.F1Score(task="multiclass", num_classes=2, average="macro")
                self.test_auroc = tm.AUROC(task="binary")
                self.test_ap = tm.AveragePrecision(task="binary")

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, _):
        out = self(**batch)
        self.log("train_loss", out.loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return out.loss

    def validation_step(self, batch, _):
        out = self(**batch)
        self.log("val_loss", out.loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if TM_AVAILABLE:
            if self.task == "regression":
                y = batch["labels"].float().view(-1)
                pred = out.logits.view(-1)
                self.val_mae.update(pred, y)
                self.val_rmse.update(pred, y)
            else:
                logits = out.logits
                y = batch["labels"].long().view(-1)
                probs1 = torch.softmax(logits, dim=-1)[:, 1]
                pred = torch.argmax(logits, dim=-1)
                self.val_acc.update(pred, y)
                self.val_f1.update(pred, y)
                self.val_auroc.update(probs1, y)
                self.val_ap.update(probs1, y)
        return out.loss

    def on_validation_epoch_end(self):
        if TM_AVAILABLE:
            if self.task == "regression":
                self.log("val_mae", self.val_mae.compute(), prog_bar=True, sync_dist=True)
                self.log("val_rmse", self.val_rmse.compute(), prog_bar=True, sync_dist=True)
                self.val_mae.reset(); self.val_rmse.reset()
            else:
                self.log("val_acc", self.val_acc.compute(), prog_bar=True, sync_dist=True)
                self.log("val_f1", self.val_f1.compute(), prog_bar=False, sync_dist=True)
                self.log("val_auroc", self.val_auroc.compute(), prog_bar=True, sync_dist=True)
                self.log("val_ap", self.val_ap.compute(), prog_bar=False, sync_dist=True)
                self.val_acc.reset(); self.val_f1.reset(); self.val_auroc.reset(); self.val_ap.reset()

    def test_step(self, batch, _):
        out = self(**batch)
        self.log("test_loss", out.loss, prog_bar=True, on_epoch=True, sync_dist=True)
        if TM_AVAILABLE:
            if self.task == "regression":
                y = batch["labels"].float().view(-1)
                pred = out.logits.view(-1)
                self.test_mae.update(pred, y)
                self.test_rmse.update(pred, y)
            else:
                logits = out.logits
                y = batch["labels"].long().view(-1)
                probs1 = torch.softmax(logits, dim=-1)[:, 1]
                pred = torch.argmax(logits, dim=-1)
                self.test_acc.update(pred, y)
                self.test_f1.update(pred, y)
                self.test_auroc.update(probs1, y)
                self.test_ap.update(probs1, y)
        return out.loss

    def on_test_epoch_end(self):
        if TM_AVAILABLE:
            if self.task == "regression":
                self.log("test_mae", self.test_mae.compute(), prog_bar=True, sync_dist=True)
                self.log("test_rmse", self.test_rmse.compute(), prog_bar=True, sync_dist=True)
                self.test_mae.reset(); self.test_rmse.reset()
            else:
                self.log("test_acc", self.test_acc.compute(), prog_bar=True, sync_dist=True)
                self.log("test_f1", self.test_f1.compute(), prog_bar=False, sync_dist=True)
                self.log("test_auroc", self.test_auroc.compute(), prog_bar=True, sync_dist=True)
                self.log("test_ap", self.test_ap.compute(), prog_bar=False, sync_dist=True)
                self.test_acc.reset(); self.test_f1.reset(); self.test_auroc.reset(); self.test_ap.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

# ------------------------- DataModule -------------------------
class YieldDataModule(pl.LightningDataModule):
    def __init__(self,
                 train: Path,
                 val: Optional[Path],
                 test: Optional[Path],
                 base_model: str,
                 task: str,
                 threshold: float,
                 normalize: bool,
                 validation_fraction: float,
                 text_col: str,
                 label_col: str,
                 max_len: int,
                 batch_size: int,
                 num_workers: int,
                 offline: bool = False):
        super().__init__()
        self.train_path = Path(train)
        self.val_path = Path(val) if val else None
        self.test_path = Path(test) if test else None

        self.base_model = base_model
        self.task = task
        self.threshold = threshold
        self.normalize = (normalize and task == "regression")
        self.validation_fraction = validation_fraction

        self.text_col = text_col
        self.label_col = label_col
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.offline = offline

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, use_fast=True, local_files_only=self.offline,
        )
        self._norm: Optional[Tuple[float, float]] = None

    def setup(self, stage: Optional[str] = None):
        tr = prepare_dataframe(load_table(self.train_path), self.text_col, self.label_col, "Train")

        if self.val_path is not None:
            va = prepare_dataframe(load_table(self.val_path), self.text_col, self.label_col, "Val")
        elif self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            tr, va = train_test_split(tr, test_size=self.validation_fraction, random_state=42, shuffle=True)
            tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
        else:
            va = None

        te = None
        if self.test_path is not None:
            te = prepare_dataframe(load_table(self.test_path), self.text_col, self.label_col, "Test")

        if self.task == "classification":
            def binarize(df: pd.DataFrame):
                df.loc[:, self.label_col] = (df[self.label_col].astype(float) >= self.threshold).astype(int)
            binarize(tr)
            if va is not None: binarize(va)
            if te is not None: binarize(te)
        else:
            if self.normalize:
                m, s = tr[self.label_col].mean(), tr[self.label_col].std()
                if s and not math.isclose(float(s), 0.0):
                    tr.loc[:, self.label_col] = (tr[self.label_col] - m) / s
                    if va is not None: va.loc[:, self.label_col] = (va[self.label_col] - m) / s
                    if te is not None: te.loc[:, self.label_col] = (te[self.label_col] - m) / s
                    self._norm = (float(m), float(s))

        self.train_df, self.val_df, self.test_df = tr, va, te

    def _make_ds(self, df: pd.DataFrame):
        return SmilesDataset(df, self.text_col, self.label_col, self.tokenizer, self.max_len, self.task)

    def train_dataloader(self):
        return DataLoader(self._make_ds(self.train_df), batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=dynamic_collate)

    def val_dataloader(self):
        if self.val_df is None: return None
        return DataLoader(self._make_ds(self.val_df), batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=dynamic_collate)

    def test_dataloader(self):
        if self.test_df is None: return None
        return DataLoader(self._make_ds(self.test_df), batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=dynamic_collate)

# ----------------------------- CLI -----------------------------
def build_argparser():
    ap = argparse.ArgumentParser("Lightning SMILES BERT (offline-ready)")

    # Note: see the surrounding code for details.
    ap.add_argument("--train", type=Path, default=PROJECT_ROOT / "data_xj" / "processed_train.tsv")
    ap.add_argument("--val",   type=Path, default=PROJECT_ROOT / "data_xj" / "processed_val.tsv")
    ap.add_argument("--test",  type=Path, default=PROJECT_ROOT / "data_xj" / "processed_test.tsv")
    ap.add_argument("--text-col", type=str, default="text")
    ap.add_argument("--label-col", type=str, default="labels")
    ap.add_argument("--validation-fraction", type=float, default=0.0)

    # Note: see the surrounding code for details.
    ap.add_argument("--base-model", type=str, default="seyonec/ChemBERTa-zinc-base-v1",
                    help='HF model name or local path.')
    ap.add_argument("--from-scratch", action="store_true", help='Do not load pretrained weights; initialize randomly')
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--offline", action="store_true", help='Fully offline: load locally and disable network access')

    # Note: see the surrounding code for details.
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)

    # Note: see the surrounding code for details.
    ap.add_argument("--accumulate", type=int, default=1)
    ap.add_argument("--precision", type=str, default="16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--strategy", type=str, default="ddp",
                    choices=["auto", "ddp", "ddp_find_unused_parameters_true", "ddp_find_unused_parameters_false"])
    ap.add_argument("--seed", type=int, default=42)

    # Note: see the surrounding code for details.
    ap.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "pl_run")
    ap.add_argument('--log-level', type=str, default='INFO', help='INFO/DEBUG/WARNING/ERROR')
    ap.add_argument("--monitor", type=str, default="val_loss")
    ap.add_argument("--mode", type=str, default="min", choices=["min", "max"])
    ap.add_argument("--patience", type=int, default=5)
    # Note: see the surrounding code for details.
    # Note: see the surrounding code for details.
    ap.add_argument("--save-top-k", type=int, default=4,
                    help='Save the top K checkpoints by monitor metric.')
    ap.add_argument("--logger", type=str, default="csv", choices=["csv", "none"])

    # Note: see the surrounding code for details.
    ap.add_argument("--resume-ckpt", type=Path, default=None, help='Resume training including optimizer state and step count')
    ap.add_argument("--load-ckpt", type=Path, default=None, help='Load model weights only for fine-tuning')
    ap.add_argument("--strict-load", action="store_true", help='Require strict weight matching')

    # Note: see the surrounding code for details.
    ap.add_argument("--no-normalize", action="store_true")
    return ap

# ----------------------------- Main -----------------------------
def main():
    args = build_argparser().parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    setup_loggers(args.outdir, args.log_level)
    pl.seed_everything(args.seed, workers=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Data
    dm = YieldDataModule(
        train=args.train,
        val=args.val,
        test=args.test,
        base_model=args.base_model,
        task=args.task,
        threshold=args.threshold,
        normalize=(not args.no_normalize),
        validation_fraction=args.validation_fraction,
        text_col=args.text_col,
        label_col=args.label_col,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        offline=args.offline,
    )
    dm.setup()

    # Model
    module = YieldBERTModule(
        model_name_or_path=args.base_model,
        task=args.task,
        lr=args.lr,
        dropout=args.dropout,
        from_scratch=args.from_scratch,
        offline=args.offline,
    )

    # Note: see the surrounding code for details.
    if args.load_ckpt is not None:
        ckpt = torch.load(args.load_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state, strict=args.strict_load)
        rank_zero_only(print)(f"[load-ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    # Callbacks & Logger
    # Note: see the surrounding code for details.
    # Note: see the surrounding code for details.
    # Note: see the surrounding code for details.
    # Note: see the surrounding code for details.
    ckpt_cb = ModelCheckpoint(
        dirpath=str(args.outdir),
        filename="best-{epoch:02d}-{"+args.monitor+":.4f}",
        monitor=args.monitor,
        mode=args.mode,
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    #es_cb = EarlyStopping(monitor=args.monitor, mode=args.mode, patience=args.patience, verbose=True)

    logger_pl = CSVLogger(save_dir=str(args.outdir), name="csv") if args.logger == "csv" else False

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else None,
        strategy=args.strategy if torch.cuda.is_available() and args.devices > 1 else "auto",
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate,
        default_root_dir=str(args.outdir),
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb],# es_cb],
        logger=logger_pl,
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(
        model=module,
        datamodule=dm,
        ckpt_path=str(args.resume_ckpt) if args.resume_ckpt is not None else None,
    )

    # Test (best ckpt)
    if dm.test_df is not None:
        trainer.test(module, datamodule=dm, ckpt_path="best")
    print(f"Done. Artifacts -> {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
