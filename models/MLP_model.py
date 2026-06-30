# -*- coding: utf-8 -*-
import os, sys, json, argparse, logging, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)


# Note: see the surrounding code for details.
logger = logging.getLogger("mlp_run")
class _Tee:
    def __init__(self, filename: str, stream):
        self.stream = stream
        self.file = open(filename, 'a', encoding='utf-8')
    def write(self, data):
        self.stream.write(data); self.file.write(data)
    def flush(self):
        self.stream.flush(); self.file.flush()

def setup_loggers(log_level: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'training.log')
    sys.stdout = _Tee(log_path, sys.stdout)
    sys.stderr = _Tee(log_path, sys.stderr)
    logger.handlers.clear()
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(stream=sys.stdout); sh.setLevel(level); sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f'Log file: {log_path}')

# Note: see the surrounding code for details.
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def binarize_y(y: np.ndarray, threshold: float) -> np.ndarray:
    return (y.astype(np.float32) > float(threshold)).astype(np.int64)

def load_features(feat_dir: str):
    paths = {k: os.path.join(feat_dir, f"{k}.npy") for k in
             ["X_train","y_train","X_val","y_val","X_test","y_test"]}
    for k,p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f'Status: {p}')
    X_tr = np.load(paths["X_train"]).astype(np.float32, copy=False)
    y_tr = np.load(paths["y_train"])
    X_va = np.load(paths["X_val"]).astype(np.float32, copy=False)
    y_va = np.load(paths["y_val"])
    X_te = np.load(paths["X_test"]).astype(np.float32, copy=False)
    y_te = np.load(paths["y_test"])
    return X_tr, y_tr, X_va, y_va, X_te, y_te

# Note: see the surrounding code for details.
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_list=(512,256), dropout=0.1, task="regression"):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden_list:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        out_dim = 1  # Note: see the surrounding code for details.
        layers += [nn.Linear(dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self.task = task
    def forward(self, x):
        return self.net(x).squeeze(1)  # [B]

# Note: see the surrounding code for details.
def train_epoch(model, loader, optimizer, device, task, class_weight=None):
    model.train()
    loss_fn = nn.MSELoss() if task=="regression" else nn.BCEWithLogitsLoss(pos_weight=class_weight)
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        if task=="regression":
            yb_t = yb.to(device).float()
        else:
            yb_t = yb.to(device).float()
        pred = model(xb)
        loss = loss_fn(pred, yb_t)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * yb_t.size(0); n += yb_t.size(0)
    return total_loss / max(n,1)

@torch.no_grad()
def eval_epoch(model, loader, device, task, threshold=0.5):
    model.eval()
    losses, n = 0.0, 0
    if task=="regression":
        loss_fn = nn.MSELoss()
        ys, ps = [], []
        for xb, yb in loader:
            xb = xb.to(device); yb_t = yb.to(device).float()
            pred = model(xb)
            loss = loss_fn(pred, yb_t)
            losses += loss.item()* yb_t.size(0); n += yb_t.size(0)
            ys.append(yb_t.cpu().numpy()); ps.append(pred.cpu().numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        mae = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))

        r2 = r2_score(y, p)
        return (losses/max(n,1)), {"mae":mae, "rmse":rmse, "r2":r2}, (y, p)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        ys, logits = [], []
        for xb, yb in loader:
            xb = xb.to(device); yb_t = yb.to(device).float()
            logit = model(xb)
            loss = loss_fn(logit, yb_t)
            losses += loss.item()* yb_t.size(0); n += yb_t.size(0)
            ys.append(yb_t.cpu().numpy()); logits.append(logit.cpu().numpy())
        y = np.concatenate(ys).astype(int)
        logit = np.concatenate(logits)
        prob = 1.0 / (1.0 + np.exp(-logit))
        yhat = (prob >= threshold).astype(int)
        # Note: see the surrounding code for details.
        auc = roc_auc_score(y, prob) if len(np.unique(y))==2 else np.nan
        ap  = average_precision_score(y, prob) if len(np.unique(y))==2 else np.nan
        acc = accuracy_score(y, yhat)
        f1  = f1_score(y, yhat, zero_division=0)
        p   = precision_score(y, yhat, zero_division=0)
        r   = recall_score(y, yhat, zero_division=0)
        return (losses/max(n,1)), {"acc":acc,"f1":f1,"prec":p,"rec":r,"auc":auc,"ap":ap}, (y, yhat, prob)

def numpy_to_loader(X, y, batch_size, shuffle):
    X_t = torch.from_numpy(X.astype(np.float32, copy=False))
    if y.dtype.kind in ('i','u'):
        y_t = torch.from_numpy(y.astype(np.int64, copy=False))
    else:
        y_t = torch.from_numpy(y.astype(np.float32, copy=False))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# Note: see the surrounding code for details.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat_dir', required=True, help='Directory containing precomputed feature files')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--task', choices=['regression','classification'], default='regression')
    ap.add_argument('--threshold', type=float, default=10.0, help='yield > threshold is treated as the positive class')
    ap.add_argument('--layers', type=str, default='256,512,512,256,128', help='Hidden layers, e.g. "512,256,128"')
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=10, help='Early-stopping patience')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-level', type=str, default='INFO')
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    setup_loggers(args.log_level, args.outdir)

    # Note: see the surrounding code for details.
    X_tr, y_tr_raw, X_va, y_va_raw, X_te, y_te_raw = load_features(args.feat_dir)

    # Note: see the surrounding code for details.
    if args.task == 'regression':
        y_tr = np.clip(y_tr_raw.astype(np.float32), 0.0, 100.0)
        y_va = np.clip(y_va_raw.astype(np.float32), 0.0, 100.0)
        y_te = np.clip(y_te_raw.astype(np.float32), 0.0, 100.0)
    else:
        y_tr = binarize_y(y_tr_raw, args.threshold)
        y_va = binarize_y(y_va_raw, args.threshold)
        y_te = binarize_y(y_te_raw, args.threshold)

    # Note: see the surrounding code for details.
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # 4) DataLoader
    train_loader = numpy_to_loader(X_tr_s, y_tr, args.batch_size, shuffle=True)
    val_loader   = numpy_to_loader(X_va_s, y_va, args.batch_size, shuffle=False)
    test_loader  = numpy_to_loader(X_te_s, y_te, args.batch_size, shuffle=False)

    # Note: see the surrounding code for details.
    hidden = tuple(int(x) for x in args.layers.split(',') if x.strip())
    model = MLP(in_dim=X_tr.shape[1], hidden_list=hidden, dropout=args.dropout, task=args.task)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Note: see the surrounding code for details.
    class_weight = None
    if args.task == 'classification':
        pos = float((y_tr==1).sum()); neg = float((y_tr==0).sum())
        if pos > 0 and neg > 0:
            class_weight = torch.tensor([neg/max(pos,1e-6)], dtype=torch.float, device=device)  # pos_weight

    # Note: see the surrounding code for details.
    best_val = float('inf')
    best_state = None
    curve_rows = []
    no_improve = 0

    logger.info(f'Status: {args.task} {device} {X_tr.shape[1]} {hidden}')
    for epoch in range(1, args.epochs+1):
        tic = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, device, args.task, class_weight)
        va_loss, va_metrics, _ = eval_epoch(model, val_loader, device, args.task)
        toc = time.time()
        # Note: see the surrounding code for details.
        if args.task=='regression':
            logger.info(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | "
                        f"val_loss={va_loss:.4f} (MAE={va_metrics['mae']:.3f}, RMSE={va_metrics['rmse']:.3f}, R2={va_metrics['r2']:.3f}) | {toc-tic:.1f}s")
            curve_rows.append([epoch, tr_loss, va_loss, va_metrics['mae'], va_metrics['rmse'], va_metrics['r2']])
        else:
            logger.info(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | "
                        f"val_loss={va_loss:.4f} (Acc={va_metrics['acc']:.3f}, F1={va_metrics['f1']:.3f}, "
                        f"P={va_metrics['prec']:.3f}, R={va_metrics['rec']:.3f}, AUC={va_metrics['auc']:.3f}, AP={va_metrics['ap']:.3f}) | {toc-tic:.1f}s")
            curve_rows.append([epoch, tr_loss, va_loss, va_metrics['acc'], va_metrics['f1'], va_metrics['prec'], va_metrics['rec'], va_metrics['auc'], va_metrics['ap']])

        # Note: see the surrounding code for details.
        #if va_loss + 1e-7 < best_val:
        #    best_val = va_loss; best_state = model.state_dict(); no_improve = 0
        #else:
        #    no_improve += 1
        #    if no_improve >= args.patience:
        # Note: see the surrounding code for details.
        #        break

    # Note: see the surrounding code for details.
    curve_path = os.path.join(args.outdir, 'training_curve.csv')
    if args.task=='regression':
        cols = ['epoch','train_loss','val_loss','val_mae','val_rmse','val_r2']
    else:
        cols = ['epoch','train_loss','val_loss','val_acc','val_f1','val_prec','val_rec','val_auc','val_ap']
    pd.DataFrame(curve_rows, columns=cols).to_csv(curve_path, index=False)
    logger.info(f'Training curve saved: {curve_path}')

    # Note: see the surrounding code for details.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Note: see the surrounding code for details.
    # Note: see the surrounding code for details.
    va_loss, va_metrics, va_dump = eval_epoch(model, val_loader, device, args.task)
    if args.task=='regression':
        y, p = va_dump
        pd.DataFrame({'y_true': y, 'y_pred': p}).to_csv(os.path.join(args.outdir, 'val_predictions.csv'), index=False)
        logger.info(f"VAL(final): loss={va_loss:.4f} | MAE={va_metrics['mae']:.3f} RMSE={va_metrics['rmse']:.3f} R2={va_metrics['r2']:.3f}")
    else:
        y, yhat, prob = va_dump
        pd.DataFrame({'y_true': y, 'y_pred': yhat, 'y_proba': prob}).to_csv(os.path.join(args.outdir, 'val_predictions.csv'), index=False)
        logger.info(f"VAL(final): loss={va_loss:.4f} | Acc={va_metrics['acc']:.3f} F1={va_metrics['f1']:.3f} "
                    f"P={va_metrics['prec']:.3f} R={va_metrics['rec']:.3f} AUC={va_metrics['auc']:.3f} AP={va_metrics['ap']:.3f}")

    # Note: see the surrounding code for details.
    te_loss, te_metrics, te_dump = eval_epoch(model, test_loader, device, args.task)
    if args.task=='regression':
        y, p = te_dump
        pd.DataFrame({'y_true': y, 'y_pred': p}).to_csv(os.path.join(args.outdir, 'test_predictions.csv'), index=False)
        logger.info(f"TEST: loss={te_loss:.4f} | MAE={te_metrics['mae']:.3f} RMSE={te_metrics['rmse']:.3f} R2={te_metrics['r2']:.3f}")
    else:
        y, yhat, prob = te_dump
        pd.DataFrame({'y_true': y, 'y_pred': yhat, 'y_proba': prob}).to_csv(os.path.join(args.outdir, 'test_predictions.csv'), index=False)
        logger.info(f"TEST: loss={te_loss:.4f} | Acc={te_metrics['acc']:.3f} F1={te_metrics['f1']:.3f} "
                    f"P={te_metrics['prec']:.3f} R={te_metrics['rec']:.3f} AUC={te_metrics['auc']:.3f} AP={te_metrics['ap']:.3f}")

    # Note: see the surrounding code for details.
    ckpt_path = os.path.join(args.outdir, 'mlp_model.pt')
    torch.save({'state_dict': model.state_dict()}, ckpt_path)
    meta = {
        'task': args.task,
        'threshold': args.threshold if args.task=='classification' else None,
        'layers': [int(x) for x in args.layers.split(',') if x],
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'seed': args.seed,
        'scaler': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()},
        'feat_dir': args.feat_dir,
        'outdir': args.outdir
    }
    with open(os.path.join(args.outdir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f'Model saved: {ckpt_path}')
    logger.info("Done.")

if __name__ == '__main__':
    main()
