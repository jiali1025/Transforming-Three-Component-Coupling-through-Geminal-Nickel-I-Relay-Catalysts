#!/usr/bin/env python3
# Note: see the surrounding code for details.
# Lightning >= 2.2

from __future__ import annotations

import importlib
import logging
import os
import time
import torch
import torch.cuda
torch.set_float32_matmul_precision('high')
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from rdkit import RDLogger

# -------------- Required paths ---------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = Path(os.getenv("TRAIN_CSV", str(PROJECT_ROOT / "splits" / "train.csv")))
VAL_CSV   = Path(os.getenv("VAL_CSV",   str(PROJECT_ROOT / "splits" / "val.csv")))
TEST_CSV  = Path(os.getenv("TEST_CSV",  str(PROJECT_ROOT / "splits" / "test.csv")))
BASE_DIR  = Path(os.getenv("BASE_DIR",  str(PROJECT_ROOT / "runs" / "run_cls_noattn")))

# -------------- Hyper-parameters -------------
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",   256))
NUM_WORKERS     = int(os.getenv("NUM_WORKERS",   4))
MAX_EPOCHS      = 100
VAL_EVERY       = int(os.getenv("VAL_EVERY",    800))
LR              = float(os.getenv("LR",         3e-4))
D_HIDDEN        = int(os.getenv("D_HIDDEN",     1024))
YIELD_THRESHOLD = float(os.getenv("YIELD_THRESHOLD", 10.0))
GPU_IDS         = [int(x) for x in os.getenv("GPU_IDS", "4,5,6,7").split(",")]

# Note: see the surrounding code for details.
# N_ATTN_LAYERS = int(os.getenv("N_LAYERS", 4))

RUN_NAME = (
    f"NiGAC_cls_noattn_bs{BATCH_SIZE}_{len(GPU_IDS)}g_"
    f"lr{LR}_h{D_HIDDEN}_thr{YIELD_THRESHOLD}_100ep"
)

# -------------- Logging setup ----------------
CKPT_DIR    = BASE_DIR / "checkpoints"
LOG_DIR     = BASE_DIR / "logs"
RUN_LOG_DIR = LOG_DIR / RUN_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RUN_LOG_DIR / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("yield_classifier_noattn")

# -------------- Silence RDKit ----------------
os.environ["RDKit_LOG_LEVEL"] = "ERROR"
RDLogger.DisableLog("rdApp.*")
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

logger.info("Run name: %s", RUN_NAME)
logger.info("Train CSV: %s", TRAIN_CSV)
logger.info("Val CSV: %s", VAL_CSV)
logger.info("Test CSV: %s", TEST_CSV)
logger.info(
    "Hyperparameters - batch_size=%d, lr=%.3e, hidden=%d, "
    "threshold=%.2f, max_epochs=%d, val_every=%d",
    BATCH_SIZE, LR, D_HIDDEN, YIELD_THRESHOLD, MAX_EPOCHS, VAL_EVERY,
)

# -------------- Data & Model ---------------
from model_dataloader import build_loader
from models.Ni_GAC_classifier_noattn import YieldClassifierNoAttn

train_loader = build_loader(TRAIN_CSV, BATCH_SIZE, NUM_WORKERS, shuffle=True)
val_loader   = build_loader(VAL_CSV,   BATCH_SIZE, NUM_WORKERS, shuffle=False)

train_batches          = max(1, len(train_loader))
effective_val_interval = min(VAL_EVERY, train_batches)
if effective_val_interval != VAL_EVERY:
    logger.info(
        "Adjusted val_check_interval from %d to %d to fit %d training batches.",
        VAL_EVERY, effective_val_interval, train_batches,
    )

# Note: see the surrounding code for details.
TOTAL_STEPS = train_batches * MAX_EPOCHS
logger.info("STEPS_PER_EPOCH=%d  TOTAL_STEPS=%d", train_batches, TOTAL_STEPS)

test_loader = None
"""
test_csv_path = Path(TEST_CSV)
if test_csv_path.exists():
    test_loader = build_loader(TEST_CSV, BATCH_SIZE, NUM_WORKERS, shuffle=False)
else:
    logger.warning(
        "Test CSV not found at %s; test phase will be skipped.",
        test_csv_path,
    )
"""

model = YieldClassifierNoAttn(
    d_h=D_HIDDEN,
    lr=LR,
    threshold=YIELD_THRESHOLD,
)


# Note: see the surrounding code for details.
class ThroughputMeter(Callback):
    def __init__(self, every: int = VAL_EVERY) -> None:
        self.every  = every
        self.t0     = None
        self.n_seen = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.t0 is None:
            self.t0 = time.time()
        self.n_seen += len(batch[-1])
        if (trainer.global_step + 1) % self.every == 0 and self.t0 is not None:
            samples_per_sec = self.n_seen / (time.time() - self.t0)
            pl_module.log(
                "samples_per_sec",
                samples_per_sec,
                prog_bar=True, logger=True, sync_dist=True,
            )
            logger.info("Step %d - %.2f samples/s", trainer.global_step + 1, samples_per_sec)
            self.t0     = time.time()
            self.n_seen = 0


# -------------- Logger ---------------
csv_logger = CSVLogger(save_dir=LOG_DIR, name=RUN_NAME)

# -------------- Callbacks ------------
ckpt_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="step{step:09d}-vAUC{val_auroc:.4f}",
    monitor="val_auroc",
    mode="max",
    save_top_k=3,
    save_last=True,
)

callbacks = [
    TQDMProgressBar(refresh_rate=10),   # Note: see the surrounding code for details.
    ckpt_cb,
    ThroughputMeter(),
]

# -------------- Trainer --------------
trainer = L.Trainer(
    accelerator="gpu",
    devices=GPU_IDS,
    strategy="ddp",
    max_epochs=MAX_EPOCHS,
    max_steps=TOTAL_STEPS,        # Note: see the surrounding code for details.
    log_every_n_steps=50,
    precision="32-true",
    gradient_clip_val=5.0,
    callbacks=callbacks,
    logger=csv_logger,
    default_root_dir=str(BASE_DIR),
)


# -------------- Main ----------------
if __name__ == "__main__":
    logger.info("Starting training with %d GPUs", len(GPU_IDS))
    trainer.fit(model, train_loader, val_loader)

    final_ckpt = CKPT_DIR / "final_full.ckpt"
    trainer.save_checkpoint(final_ckpt, weights_only=False)

    logger.info("Training complete")
    logger.info("Best checkpoint: %s", ckpt_cb.best_model_path)
    logger.info("Last checkpoint: %s", ckpt_cb.last_model_path)
    logger.info("Final checkpoint: %s", final_ckpt)
    logger.info("CSV logs: %s", csv_logger.log_dir)

    if test_loader is not None:
        def _run_test(ckpt_path: str | None, tag: str) -> None:
            if not ckpt_path:
                logger.warning("No %s checkpoint available; skipping test.", tag)
                return
            path_obj = Path(ckpt_path)
            if not path_obj.exists():
                logger.warning(
                    "%s checkpoint path does not exist (%s); skipping test.",
                    tag.capitalize(), ckpt_path,
                )
                return
            logger.info("Testing %s checkpoint: %s", tag, ckpt_path)
            metrics = trainer.test(
                ckpt_path=ckpt_path,
                dataloaders=test_loader,
                verbose=False,
            )
            if metrics:
                logger.info("Test results (%s ckpt): %s", tag, metrics[0])

        _run_test(ckpt_cb.best_model_path, "best")
        _run_test(ckpt_cb.last_model_path, "last")
