# -*- coding: utf-8 -*-

"""Random Forest binary-classification training with optional precomputed features."""

import os
import sys
import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from joblib import dump

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

# Note: see the surrounding code for details.
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

DISABLE_TQDM = False
def pbar(iterable, **kwargs):
    if DISABLE_TQDM: return iterable
    kwargs.setdefault('dynamic_ncols', True)
    kwargs.setdefault('mininterval', 0.5)
    return tqdm(iterable, **kwargs)

# Note: see the surrounding code for details.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = str(PROJECT_ROOT / "splits" / "train.csv")
VAL_CSV   = str(PROJECT_ROOT / "splits" / "val.csv")
TEST_CSV  = str(PROJECT_ROOT / "splits" / "test.csv")
BASE_DIR  = str(PROJECT_ROOT / "runs" / "RF_1015_cls")
# ----------------------------------------------------------------------

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger("rf_yield_cls")

class _Tee:
    def __init__(self, filename: str, stream):
        self.stream = stream
        self.file = open(filename, 'a', encoding='utf-8')
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()

def setup_loggers(log_level: str, base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, 'training.log')
    sys.stdout = _Tee(log_path, sys.stdout)
    sys.stderr = _Tee(log_path, sys.stderr)
    logger.handlers.clear()
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level); sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f'Log file: {log_path}')

# Note: see the surrounding code for details.
@lru_cache(maxsize=200000)
def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles: return None
    try: return Chem.MolFromSmiles(smiles)
    except Exception: return None

def split_multi_smiles(s: str) -> List[str]:
    if not s or (isinstance(s, float) and np.isnan(s)): return []
    s = str(s).strip()
    if not s: return []
    return [p.strip() for p in s.split('.') if p.strip()]

def split_reaction_smiles(rsmi: str) -> Tuple[str, str, str]:
    if '>' in rsmi:
        parts = rsmi.split('>')
        if len(parts) == 3: return parts[0], parts[1], parts[2]
    if '>>' in rsmi:
        r, p = rsmi.split('>>')
        return r, '', p
    return rsmi, '', ''

def morgan_counts_array(smiles_list: List[str], nBits: int, radius: int) -> np.ndarray:
    arr = np.zeros((nBits,), dtype=np.int32)
    gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try:
            fp_counts = gen.GetCountFingerprint(m)  # Note: see the surrounding code for details.
            for bit, count in fp_counts.GetNonzeroElements().items():
                idx = int(bit) % nBits
                arr[idx] += int(count)
        except Exception:
            continue
    return arr

@lru_cache(maxsize=500_000)
def desc_vec_from_smiles(smiles: str) -> np.ndarray:
    if not smiles: return np.zeros(10, dtype=np.float32)
    m = Chem.MolFromSmiles(smiles)
    if m is None: return np.zeros(10, dtype=np.float32)
    vals = [
        Descriptors.MolWt(m),
        Descriptors.TPSA(m),
        rdMolDescriptors.CalcNumHBA(m),
        rdMolDescriptors.CalcNumHBD(m),
        rdMolDescriptors.CalcNumRings(m),
        rdMolDescriptors.CalcNumAromaticRings(m),
        rdMolDescriptors.CalcNumRotatableBonds(m),
        rdMolDescriptors.CalcFractionCSP3(m),
        float(m.GetNumHeavyAtoms()),
        Descriptors.MolLogP(m),
    ]
    return np.array(vals, dtype=np.float32)

def sum_descriptors(smiles_list):
    if not smiles_list: return np.zeros(10, dtype=np.float32)
    acc = np.zeros(10, dtype=np.float32)
    for s in smiles_list:
        acc += desc_vec_from_smiles(s)
    return acc

def build_features(df: pd.DataFrame, nBits:int, radius:int, split_name: str="split", use_desc: bool=False):
    X_blocks, y_list = [], []
    n_total, n_bad = len(df), 0
    iterator = pbar(df.iterrows(), total=len(df), desc=f"Featurizing[{split_name}]")
    for _, row in iterator:
        rsmi = str(row['reaction_smiles']).strip()
        y = row['yield']
        try: y = float(y)
        except Exception: y = np.nan
        if not rsmi or np.isnan(y):
            n_bad += 1; continue
        r_str, a_str, p_str = split_reaction_smiles(rsmi)
        reactants = split_multi_smiles(r_str)
        products  = split_multi_smiles(p_str)
        if len(reactants)==0 or len(products)==0:
            n_bad += 1; continue
        fp_r = morgan_counts_array(reactants, nBits, radius)
        fp_p = morgan_counts_array(products,  nBits, radius)
        fp_delta = (fp_p - fp_r).astype(np.int16)
        cats = split_multi_smiles(row.get('transition_metal_catalyst', ''))
        reag = split_multi_smiles(row.get('other_reagent', ''))
        fp_cat  = morgan_counts_array(cats, nBits, radius).astype(np.int16)
        fp_reag = morgan_counts_array(reag, nBits, radius).astype(np.int16)
        cat_missing  = 1.0 if len(cats)==0 else 0.0
        reag_missing = 1.0 if len(reag)==0 else 0.0
        feat_parts = [fp_delta.astype(np.float32), fp_cat.astype(np.float32), fp_reag.astype(np.float32)]
        if use_desc:
            desc_delta = (sum_descriptors(products) - sum_descriptors(reactants)).astype(np.float32)
            feat_parts.append(desc_delta)
        feat_parts.append(np.array([cat_missing, reag_missing], dtype=np.float32))
        X_blocks.append(np.concatenate(feat_parts))
        y_list.append(np.clip(y, 0.0, 100.0))
    if n_bad>0:
        logger.warning(f'Invalid or missing rows discarded: {n_bad} {n_total}')
    base_dim = nBits*3 + (10 if use_desc else 0) + 2
    X = np.vstack(X_blocks) if X_blocks else np.zeros((0, base_dim), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    stats = {'n_total': n_total, 'n_bad': n_bad, 'n_used': X.shape[0], 'n_features': X.shape[1] if X.shape[0] else base_dim}
    return X, y, stats

# Note: see the surrounding code for details.
def binarize_y(y: np.ndarray, threshold: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return (y > float(threshold)).astype(np.int32)

# Note: see the surrounding code for details.
def hyperparam_search(base_params: Dict[str, Any],
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray,   y_val: np.ndarray,
                      n_iter_search: int,
                      random_state: int,
                      n_jobs: int,
                      out_dir: str) -> Dict[str, Any]:
    logger.info('Starting RandomizedSearchCV with the validation split fixed as the validation fold.')
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    test_fold = np.array([-1]*len(y_train) + [0]*len(y_val))  # -1=train, 0=val
    ps = PredefinedSplit(test_fold=test_fold)

    param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [12, 16, 24],        # Note: see the surrounding code for details.
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf':  [1, 2, 3],
        'max_features': ['sqrt', 'log2', 0.2, 0.3],
        'max_samples':  [0.6, 0.8],
        'class_weight': [None, 'balanced'],  # Note: see the surrounding code for details.
    }

    scoring = {
        'roc_auc': 'roc_auc',
        'ap': 'average_precision',
        'f1': 'f1',
        'bal_acc': 'balanced_accuracy',
    }

    # Note: see the surrounding code for details.
    clf_est = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **base_params)

    search = RandomizedSearchCV(
        estimator=clf_est,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring=scoring,
        n_jobs=1,                 # Note: see the surrounding code for details.
        cv=ps.split(),
        random_state=random_state,
        verbose=1,
        refit=False,
        return_train_score=True,
    )
    search.fit(X_tv, y_tv)

    # Note: see the surrounding code for details.
    res = pd.DataFrame(search.cv_results_)
    out = pd.DataFrame()
    # Note: see the surrounding code for details.
    mapping = [
        ('split0_train_roc_auc', 'train_roc_auc', False),
        ('split0_test_roc_auc',  'val_roc_auc',   False),
        ('split0_train_ap',      'train_ap',      False),
        ('split0_test_ap',       'val_ap',        False),
        ('split0_train_f1',      'train_f1',      False),
        ('split0_test_f1',       'val_f1',        False),
        ('split0_train_bal_acc', 'train_bal_acc', False),
        ('split0_test_bal_acc',  'val_bal_acc',   False),
    ]
    for src, dst, _ in mapping:
        if src in res.columns:
            out[dst] = res[src]

    param_cols = [c for c in res.columns if c.startswith('param_')]
    out = pd.concat([res[param_cols], out], axis=1)

    # Note: see the surrounding code for details.
    rank_key = None
    if 'val_ap' in out:       rank_key = 'val_ap'
    elif 'val_roc_auc' in out: rank_key = 'val_roc_auc'
    elif 'val_f1' in out:     rank_key = 'val_f1'
    else:
        raise RuntimeError('Invalid input.')

    best_idx = int(np.nanargmax(out[rank_key].values))
    best_row = out.iloc[best_idx]
    best_params = {k.replace('param_', ''): best_row[k] for k in param_cols}

    csv_path = os.path.join(out_dir, 'search_results.csv')
    out.to_csv(csv_path, index=False)
    logger.info(f'Hyperparameter search results saved: {csv_path}')
    logger.info(f'Best parameters: {rank_key} {best_params}')

    return best_params

# Note: see the surrounding code for details.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, help='Directory containing precomputed feature files')
    parser.add_argument('--train', default=TRAIN_CSV, type=str)
    parser.add_argument('--val',   default=VAL_CSV,   type=str)
    parser.add_argument('--test',  default=TEST_CSV,  type=str)
    parser.add_argument('--outdir', default=BASE_DIR, type=str)

    # Note: see the surrounding code for details.
    parser.add_argument('--bits', type=int, default=128)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--use-desc', action='store_true')

    # Note: see the surrounding code for details.
    parser.add_argument('--n-estimators', type=int, default=600, dest='n_estimators')
    parser.add_argument('--max-depth', type=int, default=None, dest='max_depth')
    parser.add_argument('--random-state', type=int, default=42, dest='random_state')
    parser.add_argument('--n-jobs', type=int, default=-1, dest='n_jobs')
    parser.add_argument('--n-iter-search', type=int, default=25, dest='n_iter_search')
    parser.add_argument('--no-refit', action='store_true')

    # Note: see the surrounding code for details.
    parser.add_argument('--threshold', type=float, default=10.0, help='yield > threshold is treated as the positive class')

    # Note: see the surrounding code for details.
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--no-progress', action='store_true')

    args = parser.parse_args()
    global DISABLE_TQDM
    DISABLE_TQDM = args.no_progress

    os.makedirs(args.outdir, exist_ok=True)
    setup_loggers(args.log_level, args.outdir)

    # Note: see the surrounding code for details.
    if args.feat_dir:
        print('==== Status ====')
        feat_files = [('train','X_train.npy','y_train.npy'),
                      ('val',  'X_val.npy',  'y_val.npy'),
                      ('test', 'X_test.npy', 'y_test.npy')]
        data = {}
        for split_name, x_file, y_file in feat_files:
            x_path = os.path.join(args.feat_dir, x_file)
            y_path = os.path.join(args.feat_dir, y_file)
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(f'Status: {x_path} {y_path}')
            data[f'X_{split_name}'] = np.load(x_path)
            data[f'y_{split_name}'] = np.load(y_path)
            print(f"{split_name}: X.shape={data[f'X_{split_name}'].shape}, y.shape={data[f'y_{split_name}'].shape}")

        # Note: see the surrounding code for details.
        for split in ['train','val','test']:
            Xk = f'X_{split}'
            if data[Xk].dtype != np.float32:
                data[Xk] = data[Xk].astype(np.float32, copy=False)
            Yk = f'y_{split}'
            if data[Yk].dtype != np.float32:
                data[Yk] = data[Yk].astype(np.float32, copy=False)

        # Note: see the surrounding code for details.
        y_tr = binarize_y(data['y_train'], args.threshold)
        y_va = binarize_y(data['y_val'],   args.threshold)
        y_te = binarize_y(data['y_test'],  args.threshold)

        X_tr, X_va, X_te = data['X_train'], data['X_val'], data['X_test']
        st_tr = {'n_used': X_tr.shape[0], 'n_features': X_tr.shape[1]}
        st_va = {'n_used': X_va.shape[0], 'n_features': X_va.shape[1]}
        st_te = {'n_used': X_te.shape[0], 'n_features': X_te.shape[1]}

    else:
        print('==== Status ====')
        print(f"train: {args.train}\nval:   {args.val}\ntest:  {args.test}")
        required_cols = {'reaction_smiles','transition_metal_catalyst','other_reagent','yield'}
        df_tr = pd.read_csv(args.train, sep=None, engine='python')
        df_va = pd.read_csv(args.val,   sep=None, engine='python')
        df_te = pd.read_csv(args.test,  sep=None, engine='python')
        for name, df in [('train', df_tr), ('val', df_va), ('test', df_te)]:
            missing = required_cols - set(df.columns)
            if missing: raise ValueError(f'Missing required columns: {name} {missing}')
        X_tr, y_tr_f, st_tr = build_features(df_tr, args.bits, args.radius, 'train', args.use_desc)
        X_va, y_va_f, st_va = build_features(df_va, args.bits, args.radius, 'val',   args.use_desc)
        X_te, y_te_f, st_te = build_features(df_te, args.bits, args.radius, 'test',  args.use_desc)
        # Note: see the surrounding code for details.
        y_tr = binarize_y(y_tr_f, args.threshold)
        y_va = binarize_y(y_va_f, args.threshold)
        y_te = binarize_y(y_te_f, args.threshold)

    logger.info(f"Feature summary: {st_tr['n_features']} {args.use_desc} {st_tr['n_used']} {st_va['n_used']} {st_te['n_used']}")
    logger.info(f'Label summary: {y_tr.sum()} {len(y_tr)} {y_va.sum()} {len(y_va)} {y_te.sum()} {len(y_te)} {args.threshold}')

    # Note: see the surrounding code for details.
    base_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'bootstrap': True,
        # Note: see the surrounding code for details.
    }

    # Note: see the surrounding code for details.
    best_params = hyperparam_search(
        base_params,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        n_iter_search=args.n_iter_search,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        out_dir=args.outdir,
    )

    # Note: see the surrounding code for details.
    clf_train = RandomForestClassifier(random_state=args.random_state, n_jobs=args.n_jobs, **best_params)
    clf_train.fit(X_tr, y_tr)
    va_proba = clf_train.predict_proba(X_va)[:,1]
    va_pred  = (va_proba >= 0.5).astype(int)
    def safe_auc(y_true, p):
        return roc_auc_score(y_true, p) if len(np.unique(y_true))==2 else np.nan
    def safe_ap(y_true, p):
        return average_precision_score(y_true, p) if len(np.unique(y_true))==2 else np.nan
    logger.info(
        "VAL(train-fit): "
        f"Acc={accuracy_score(y_va, va_pred):.3f} "
        f"F1={f1_score(y_va, va_pred, zero_division=0):.3f} "
        f"P={precision_score(y_va, va_pred, zero_division=0):.3f} "
        f"R={recall_score(y_va, va_pred, zero_division=0):.3f} "
        f"ROC-AUC={safe_auc(y_va, va_proba):.3f} "
        f"AP={safe_ap(y_va, va_proba):.3f}"
    )

    # Note: see the surrounding code for details.
    if not args.no_refit:
        logger.info('Refitting the final model on train+val.')
        final_params = dict(best_params)
        # Note: see the surrounding code for details.
        final_params['n_estimators'] = max(int(final_params.get('n_estimators', 100)), 600)
        clf_final = RandomForestClassifier(random_state=args.random_state, n_jobs=args.n_jobs, **final_params)
        X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
        clf_final.fit(X_tv, y_tv)
    else:
        logger.info('Do not refit the final model on train+val')
        clf_final = clf_train

    # Note: see the surrounding code for details.
    VAL_CSV_PATH = os.path.join(args.outdir, 'val_predictions.csv')
    va_proba_f = clf_final.predict_proba(X_va)[:,1]
    va_pred_f  = (va_proba_f >= 0.5).astype(int)
    pd.DataFrame({'y_true': y_va, 'y_pred': va_pred_f, 'y_proba': va_proba_f}).to_csv(VAL_CSV_PATH, index=False)
    logger.info(
        "VAL(final): "
        f"Acc={accuracy_score(y_va, va_pred_f):.3f} "
        f"F1={f1_score(y_va, va_pred_f, zero_division=0):.3f} "
        f"P={precision_score(y_va, va_pred_f, zero_division=0):.3f} "
        f"R={recall_score(y_va, va_pred_f, zero_division=0):.3f} "
        f"ROC-AUC={safe_auc(y_va, va_proba_f):.3f} "
        f"AP={safe_ap(y_va, va_proba_f):.3f}"
    )
    logger.info(f'Path: {VAL_CSV_PATH}')

    TEST_CSV_PATH = os.path.join(args.outdir, 'test_predictions.csv')
    te_proba = clf_final.predict_proba(X_te)[:,1]
    te_pred  = (te_proba >= 0.5).astype(int)
    pd.DataFrame({'y_true': y_te, 'y_pred': te_pred, 'y_proba': te_proba}).to_csv(TEST_CSV_PATH, index=False)
    logger.info(
        "TEST: "
        f"Acc={accuracy_score(y_te, te_pred):.3f} "
        f"F1={f1_score(y_te, te_pred, zero_division=0):.3f} "
        f"P={precision_score(y_te, te_pred, zero_division=0):.3f} "
        f"R={recall_score(y_te, te_pred, zero_division=0):.3f} "
        f"ROC-AUC={safe_auc(y_te, te_proba):.3f} "
        f"AP={safe_ap(y_te, te_proba):.3f}"
    )
    logger.info(f'Path: {TEST_CSV_PATH}')

    # Note: see the surrounding code for details.
    MODEL_PATH = os.path.join(args.outdir, 'model_final.joblib')
    meta = {
        'task': 'binary_classification',
        'threshold': args.threshold,
        'bits': args.bits,
        'radius': args.radius,
        'feature_blocks': {
            'delta_fp': args.bits,
            'catalyst_fp': args.bits,
            'reagent_fp': args.bits,
            'delta_desc': 10 if args.use_desc else 0,
            'flags': 2,
        },
        'random_state': args.random_state,
        'refit_on_trainval': (not args.no_refit),
        'use_descriptors': args.use_desc,
        'best_params': best_params,
        'sizes': {'train': len(y_tr), 'val': len(y_va), 'test': len(y_te)},
        'paths': {
            'feat_dir': args.feat_dir,
            'train_csv': args.train if not args.feat_dir else None,
            'val_csv': args.val if not args.feat_dir else None,
            'test_csv': args.test if not args.feat_dir else None,
            'base_dir': args.outdir,
            'search_results_csv': os.path.join(args.outdir, 'search_results.csv'),
            'val_predictions_csv': VAL_CSV_PATH,
            'test_predictions_csv': TEST_CSV_PATH,
            'log_file': os.path.join(args.outdir, 'training.log'),
        }
    }
    dump({'model': clf_final, 'meta': meta}, MODEL_PATH)
    logger.info(f'Model saved: {MODEL_PATH}')

if __name__ == '__main__':
    main()
