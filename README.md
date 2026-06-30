# Ni-GAC Models — Reaction Yield Prediction for Geminal Nickel(I) Relay Catalysis

Machine-learning code for predicting and analyzing reaction yields in
three-component coupling reactions, with a focus on transition-metal
(in particular nickel) catalyzed transformations.

This repository accompanies the paper:

> **Transforming Three-Component Coupling through Geminal Nickel(I) Relay Catalysts**

---

## Overview

This project provides the data-processing and modeling code used to study
reaction-yield prediction for catalytic three-component couplings. Given a
reaction (encoded as a reaction SMILES) together with its transition-metal
catalyst and other reagents, the models predict the reaction **yield**, either
as a continuous value (regression) or as a high/low-yield label
(classification).

The repository contains several complementary components:

- **A graph-neural-network model (`YieldPredictor` / `YieldClassifier`)** built
  on top of a vendored copy of [Chemprop](https://github.com/chemprop/chemprop)
  (v2.2.0). It encodes the reaction as a *Condensed Graph of Reaction* (CGR),
  encodes the catalyst and other reagents as molecular graphs, and fuses them
  through a multi-layer bidirectional **cross-graph attention** mechanism.
- **Baseline models**: a Random Forest and a feed-forward MLP operating on
  Morgan-fingerprint features, plus an optional SMILES-based BERT baseline.
- **Data-processing utilities** for cleaning, filtering, validating, and
  splitting the reaction dataset.

> **Note on scope.** Most training scripts were developed and run on an HPC
> cluster and contain **hard-coded absolute data paths**. Before running any
> training script you will need to edit those paths (or set the documented
> environment variables) to point at your own data. See

---

## Environment & Installation

### Python version

The vendored Chemprop is **v2.2.0** and the compiled caches in the repository
were produced with **CPython 3.12**. We recommend **Python 3.11 or 3.12**.

### Dependencies

There is **no `requirements.txt`, `environment.yml`, or `pyproject.toml`** in
the repository. The following dependency list was inferred from the imports in
the code; **please verify exact versions for your setup before relying on it.**

Core (graph models):

- `torch`
- `lightning` (PyTorch Lightning ≥ 2.2)
- `torchmetrics`
- `rdkit`
- `numpy`, `pandas`

Baselines / utilities:

- `scikit-learn`, `joblib` (Random Forest, MLP scaling/metrics)
- `tqdm`
- `matplotlib` (optional; yield-distribution plot)
- `rich` (optional; nicer progress bar)
- `transformers` (only for `train_pl_bert.py`)
- `python-pptx` (only for `build_model_blockdiagram.py`)

`chemprop` itself is **vendored** under `models/chemprop/` and is imported
directly; you do not need to `pip install chemprop`. However, Chemprop's own
runtime dependencies (RDKit, PyTorch, Lightning, etc.) must be present.

Example installation (adjust as needed):

```bash
conda create -n ni_gac python=3.12
conda activate ni_gac

# RDKit is most reliably installed via conda
conda install -c conda-forge rdkit

# PyTorch: follow https://pytorch.org for the right CUDA build
pip install torch

pip install lightning torchmetrics numpy pandas scikit-learn joblib tqdm \
            matplotlib rich
# Optional baselines:
pip install transformers python-pptx
```

> **Import-path note.** The training scripts mix package-style imports
> (`from models.Ni_GAC_model import YieldPredictor`) with flat imports
> (`from model_dataloader import build_loader`, `from combined_featurizers import ...`).
> In practice this means **both the repository root and the `models/`
> directory must be on `PYTHONPATH`**. A simple approach is to run training
> scripts from inside `models/` while adding the repository root, e.g.:
> ```bash
> cd models
> PYTHONPATH="..:.:$PYTHONPATH" python train_yield.py
> ```
> Please verify the import setup in your environment before running.

---

## Usage

> All training scripts below contain hard-coded `TRAIN_CSV` / `VAL_CSV` /
> `TEST_CSV` / `BASE_DIR` constants near the top. **Edit these before
> running.** Many hyperparameters are read from environment variables (see
> each script). *Please verify the arguments before running.*

### Inspect the data

```bash
# Yield distribution + histogram (writes train_y_hist.png)
python train_yield_distribution.py --csv splits/train.csv --clip01

# Distribution of #molecules in other_reagent (edit CSV path in the file)
python models/preprocess/stat_of_data.py
```

### Graph model — regression (`YieldPredictor`)

Edit the paths in `models/train_yield.py`, then:

```bash
cd models
# Hyperparameters can be overridden via environment variables:
BATCH_SIZE=256 LR=3e-4 D_HIDDEN=1024 N_LAYERS=4 GPU_IDS=0,1,2,3 \
  python train_yield.py
```

Key behavior:

- Multi-GPU **DDP** training via PyTorch Lightning (`GPU_IDS` selects devices).
- `OneCycleLR` schedule, `AdamW`, MSE loss; output is squashed to `[0, 100]`
  via `100 * sigmoid(.)`.
- **Automatic resume**: if a checkpoint already exists in `BASE_DIR/checkpoints`,
  training resumes from the most recent one.
- Checkpoints (top-3 by `val_loss` + `last` + `final_full.ckpt`) and CSV logs
  are written under `BASE_DIR`.

### Graph model — classification (`YieldClassifier`)

```bash
cd models
BATCH_SIZE=256 LR=3e-4 D_HIDDEN=1024 N_LAYERS=4 \
  YIELD_THRESHOLD=10.0 GPU_IDS=0,1,2,3 \
  python train_yield_classifier.py
```

Reactions are labeled high-yield when `yield >= YIELD_THRESHOLD` (default
`10.0`). Training monitors validation **AUROC**, with early stopping, and logs
accuracy / average precision / AUROC / F1.

### Ablations (no cross-attention)

```bash
cd models
python train_yield_noattn.py             # regression ablation
python train_yield_classifier_noattn.py  # classification ablation
```

### Random Forest baseline

```bash
# Step 1: precompute Morgan-fingerprint features (per split)
python models/RF_feature.py --csv splits/train.csv --split_name train \
    --save-dir RF_feature --bits 128 --radius 2
python models/RF_feature.py --csv splits/val.csv   --split_name val \
    --save-dir RF_feature --bits 128 --radius 2
python models/RF_feature.py --csv splits/test.csv  --split_name test \
    --save-dir RF_feature --bits 128 --radius 2

# Step 2: train the Random Forest regressor on precomputed features
python models/RF_model.py --feat_dir RF_feature --outdir RF_run --n-iter-search 50
```

Notes:
- Features are `[Δfingerprint, catalyst_fp, reagent_fp, (optional descriptors), 2 missing-flags]`.
  With `--bits 128` and no descriptors this is `3*128 + 2 = 386` dimensions.
- `RF_model.py` can also build features on the fly from CSVs (`--train/--val/--test`)
  if `--feat_dir` is omitted. A hyperparameter search block exists but is
  commented out; a fixed best-parameter set is used by default.
  *Please verify the arguments before running.*
- `RF_model_classifier.py` provides the classification counterpart.

### MLP baseline

Operates on the same precomputed feature `.npy` files:

```bash
python models/MLP_model.py --feat_dir RF_feature --outdir MLP_run \
    --task regression --epochs 80 --layers 256,512,512,256,128
# Classification:
python models/MLP_model.py --feat_dir RF_feature --outdir MLP_cls_run \
    --task classification --threshold 10.0
```

### Optional: SMILES BERT baseline

`models/train_pl_bert.py` fine-tunes a HuggingFace sequence-classification
model on reaction SMILES. It is argparse-driven; inspect the script for its
options. *Please verify the arguments before running.*

---
