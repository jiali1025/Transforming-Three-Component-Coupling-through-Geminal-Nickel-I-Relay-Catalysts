#!/usr/bin/env python3
"""Return a DataLoader for graph batches and targets."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List

import lightning as L

try:
    from models.Ni_GAC_classifier import YieldClassifier
    from models.model_dataloader import build_loader
except ImportError:
    from Ni_GAC_classifier import YieldClassifier
    from model_dataloader import build_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_CSV = PROJECT_ROOT / "splits" / "test.csv"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "runs" / "run_cls_v1" / "checkpoints"


def _build_trainer(devices: Iterable[int]) -> L.Trainer:
    return L.Trainer(
        accelerator="gpu",
        devices=list(devices),
        precision="32-true",
        logger=False,
        enable_progress_bar=True,
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    keys = ["test_acc", "test_auroc", "test_f1", "test_ap", "test_loss"]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key, value in metrics.items():
        if key not in keys:
            parts.append(f"{key}={value:.4f}")
    return " | ".join(parts)


def run_test(ckpt_path: Path, devices: Iterable[int], test_loader) -> None:
    if not ckpt_path.is_file():
        logging.warning("Checkpoint not found: %s", ckpt_path)
        return

    logging.info("Loading checkpoint: %s", ckpt_path)
    model = YieldClassifier.load_from_checkpoint(str(ckpt_path), map_location="cpu")

    trainer = _build_trainer(devices)
    logging.info("Running test...")
    results: List[dict] = trainer.test(model=model, dataloaders=test_loader, verbose=False)
    if not results:
        logging.warning("No metrics returned for %s", ckpt_path)
        return

    summary = _format_metrics(results[0])
    print(f"[{ckpt_path.name}] {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Ni-GAC yield classifier checkpoints.")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help="Directory containing final_full.ckpt / last.ckpt.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Path to a single checkpoint file; overrides --ckpt-dir if provided.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=os.getenv("GPU_IDS", "0"),
        help="Comma-separated list of GPU device ids to use, e.g. '0' or '0,1'.",
    )
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 256)))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", 8)))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    device_ids = [int(x) for x in args.devices.split(",") if x.strip()]

    test_loader = build_loader(args.test_csv, args.batch_size, args.num_workers, shuffle=False)

    ckpt_paths: List[Path]
    if args.ckpt:
        ckpt_paths = [args.ckpt]
    else:
        ckpt_paths = [
            args.ckpt_dir / "final_full.ckpt",
            args.ckpt_dir / "last.ckpt",
        ]

    for ckpt_path in ckpt_paths:
        run_test(ckpt_path, device_ids, test_loader)


if __name__ == "__main__":
    main()
