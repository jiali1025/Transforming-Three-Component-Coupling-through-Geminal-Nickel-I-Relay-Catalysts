#!/usr/bin/env python3
# Note: see the surrounding code for details.
# Lightning >= 2.2
from __future__ import annotations
import os, sys, time, logging, importlib, traceback
from datetime import datetime
from pathlib import Path
import torch
import torch.cuda
torch.set_float32_matmul_precision('high')
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, Callback, TQDMProgressBar
)
from lightning.pytorch.loggers import CSVLogger
from rdkit import RDLogger


# Note: see the surrounding code for details.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = Path(os.getenv("TRAIN_CSV", str(PROJECT_ROOT / "splits" / "train.csv")))
VAL_CSV   = Path(os.getenv("VAL_CSV",   str(PROJECT_ROOT / "splits" / "val.csv")))
BASE_DIR  = Path(os.getenv("BASE_DIR",  str(PROJECT_ROOT / "runs" / "run_noattn_0524")))
# ----------------------------------------------------------------------

CKPT_DIR = BASE_DIR / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR  = BASE_DIR / "logs";        LOG_DIR.mkdir(parents=True, exist_ok=True)

# Note: see the surrounding code for details.
LOG_FILE = BASE_DIR / "run.log"
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, x):  [f.write(x) and f.flush() for f in self.files]
    def flush(self):     [f.flush() for f in self.files]
_log_fh = open(LOG_FILE, "a", buffering=1)
_log_fh.write(f"\n===== Run started @ {datetime.now()} =====\n")
sys.stdout = Tee(sys.__stdout__, _log_fh)
sys.stderr = Tee(sys.__stderr__, _log_fh)
print(f'Log file: {LOG_FILE}')

# Note: see the surrounding code for details.
os.environ["RDKit_LOG_LEVEL"] = "ERROR"
RDLogger.DisableLog("rdApp.*")
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

# Note: see the surrounding code for details.
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 256))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
LR          = float(os.getenv("LR", 3e-4))
D_HIDDEN    = int(os.getenv("D_HIDDEN", 1024))
GPU_IDS     = [int(x) for x in os.getenv("GPU_IDS", "0,1,2,3,4,5,6,7").split(",")]
MAX_EPOCHS  = 100

# Note: see the surrounding code for details.
# N_LAYERS  = int(os.getenv("N_LAYERS", 4))

RUN_NAME = (f"NiGAC_noattn_bs{BATCH_SIZE}_{len(GPU_IDS)}g_"
            f"lr{LR}_h{D_HIDDEN}_100ep")

# Note: see the surrounding code for details.
from model_dataloader import build_loader
from models.Ni_GAC_model_noattn import YieldPredictorNoAttn

train_loader = build_loader(TRAIN_CSV, BATCH_SIZE, NUM_WORKERS, shuffle=True)
val_loader   = build_loader(VAL_CSV,   BATCH_SIZE, NUM_WORKERS, shuffle=False)

model = YieldPredictorNoAttn(d_h=D_HIDDEN, lr=LR)

# Note: see the surrounding code for details.
class ThroughputMeter(Callback):
    def on_train_epoch_start(self, *_):
        self.t0, self.seen = time.time(), 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.seen += len(batch[-1])

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(
            "samples_per_sec",
            self.seen / (time.time() - self.t0 + 1e-9),
            prog_bar=True,
        )

# Note: see the surrounding code for details.
class WhyStop(Callback):
    def on_train_batch_end(self, tr, *_):
        if tr.should_stop:
            print(f"\n[WhyStop] should_stop=True epoch={tr.current_epoch} step={tr.global_step}")
            print("trainer.state:", tr.state.__dict__, flush=True)

# Logger and checkpoint setup.
csv_logger = CSVLogger(save_dir=LOG_DIR, name=RUN_NAME)
ckpt_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="s{step:09d}-vl{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True,
)

callbacks = [
    TQDMProgressBar(refresh_rate=10),
    ckpt_cb,
    ThroughputMeter(),
    WhyStop(),
]

# Note: see the surrounding code for details.
STEPS_PER_EPOCH = len(train_loader)
TOTAL_STEPS     = STEPS_PER_EPOCH * MAX_EPOCHS

# Trainer setup.
trainer = L.Trainer(
    accelerator="gpu",
    devices=GPU_IDS,
    strategy="ddp",
    max_epochs=MAX_EPOCHS,
    max_steps=TOTAL_STEPS,
    val_check_interval=800,
    log_every_n_steps=50,
    precision="32-true",
    callbacks=callbacks,
    logger=csv_logger,
    default_root_dir=str(BASE_DIR),
)

# Note: see the surrounding code for details.
if __name__ == "__main__":
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print('Caught exception:', repr(e))
        traceback.print_exc()
        sys.exit(1)

    final_ckpt = CKPT_DIR / "final_full.ckpt"
    trainer.save_checkpoint(final_ckpt, weights_only=False)

    print('Training completed normally.')
    print('Best ckpt:', ckpt_cb.best_model_path)
    print('Last ckpt:', ckpt_cb.last_model_path)
    print('Final ckpt:', final_ckpt)
    print('Log directory:', csv_logger.log_dir)
