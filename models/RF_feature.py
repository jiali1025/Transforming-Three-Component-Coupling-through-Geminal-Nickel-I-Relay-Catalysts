# -*- coding: utf-8 -*-
"""Extract and save RF feature arrays with resume support."""
"""
python RF_feature.py --csv ./splits/train.csv --split_name train --save-dir ./RF_feature --bits 128 --radius 2

# Add --use-desc to enable descriptor features.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from functools import lru_cache
from typing import List, Tuple, Optional

# Note: see the surrounding code for details.
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Note: see the surrounding code for details.
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Note: see the surrounding code for details.
@lru_cache(maxsize=200000)
def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def split_multi_smiles(s: str) -> List[str]:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    return [p.strip() for p in s.split('.') if p.strip()]

def split_reaction_smiles(rsmi: str) -> Tuple[str, str, str]:
    if '>' in rsmi:
        parts = rsmi.split('>')
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    if '>>' in rsmi:
        r, p = rsmi.split('>>')
        return r, '', p
    return rsmi, '', ''

def morgan_counts_array(smiles_list: List[str], nBits: int, radius: int) -> np.ndarray:
    arr = np.zeros((nBits,), dtype=np.int32)
    gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            fp_counts = gen.GetCountFingerprint(m)
            for bit, count in fp_counts.GetNonzeroElements().items():
                idx = int(bit) % nBits
                arr[idx] += int(count)
        except Exception:
            continue
    return arr

@lru_cache(maxsize=500_000)
def desc_vec_from_smiles(smiles: str) -> np.ndarray:
    if not smiles:
        return np.zeros(10, dtype=np.float32)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(10, dtype=np.float32)
    vals = [
        Descriptors.MolWt(m), Descriptors.TPSA(m),
        rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumHBD(m),
        rdMolDescriptors.CalcNumRings(m), rdMolDescriptors.CalcNumAromaticRings(m),
        rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),
        float(m.GetNumHeavyAtoms()), Descriptors.MolLogP(m),
    ]
    return np.array(vals, dtype=np.float32)

def sum_descriptors(smiles_list):
    if not smiles_list:
        return np.zeros(10, dtype=np.float32)
    acc = np.zeros(10, dtype=np.float32)
    for s in smiles_list:
        acc += desc_vec_from_smiles(s)
    return acc

# Note: see the surrounding code for details.
def build_features_with_resume(
    df: pd.DataFrame, nBits: int, radius: int, 
    save_dir: str, split_name: str, use_desc: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """English documentation for build_features_with_resume."""
    # Note: see the surrounding code for details.
    x_part_path = os.path.join(save_dir, f"X_{split_name}_part.npy")
    y_part_path = os.path.join(save_dir, f"y_{split_name}_part.npy")
    done_idx_path = os.path.join(save_dir, f"done_idx_{split_name}.txt")  # Note: see the surrounding code for details.
    final_x_path = os.path.join(save_dir, f"X_{split_name}.npy")
    final_y_path = os.path.join(save_dir, f"y_{split_name}.npy")

    # Note: see the surrounding code for details.
    if os.path.exists(final_x_path) and os.path.exists(final_y_path):
        print(f'Feature file exists; loading: {split_name} {final_x_path}')
        return np.load(final_x_path), np.load(final_y_path)
    
    # Note: see the surrounding code for details.
    start_idx = 0
    X_blocks, y_list = [], []
    if os.path.exists(done_idx_path) and os.path.exists(x_part_path) and os.path.exists(y_part_path):
        with open(done_idx_path, 'r') as f:
            start_idx = int(f.read().strip()) + 1  # Note: see the surrounding code for details.
        X_blocks = [np.load(x_part_path)]
        y_list = list(np.load(y_part_path))
        print(f'Found resume point: {start_idx} {split_name}')

    # Note: see the surrounding code for details.
    base_dim = nBits * 3 + (10 if use_desc else 0) + 2

    # Note: see the surrounding code for details.
    iterator = tqdm(
        df.iloc[start_idx:].iterrows(), 
        total=len(df) - start_idx, 
        desc=f"Featurizing[{split_name}] (from {start_idx})"
    )

    for row_idx, (_, row) in enumerate(iterator, start=start_idx):
        rsmi = str(row['reaction_smiles']).strip()
        y = row['yield']
        
        # Note: see the surrounding code for details.
        try:
            y = float(y)
        except Exception:
            continue
        if not rsmi or np.isnan(y):
            continue
        
        # Note: see the surrounding code for details.
        r_str, a_str, p_str = split_reaction_smiles(rsmi)
        reactants = split_multi_smiles(r_str)
        products = split_multi_smiles(p_str)
        if len(reactants) == 0 or len(products) == 0:
            continue

        fp_r = morgan_counts_array(reactants, nBits, radius)
        fp_p = morgan_counts_array(products, nBits, radius)
        fp_delta = (fp_p - fp_r).astype(np.int16)

        cats = split_multi_smiles(row.get('transition_metal_catalyst', ''))
        reag = split_multi_smiles(row.get('other_reagent', ''))
        fp_cat = morgan_counts_array(cats, nBits, radius).astype(np.int16)
        fp_reag = morgan_counts_array(reag, nBits, radius).astype(np.int16)
        cat_missing = 1.0 if len(cats) == 0 else 0.0
        reag_missing = 1.0 if len(reag) == 0 else 0.0

        # Note: see the surrounding code for details.
        feat_parts = [
            fp_delta.astype(np.float32),
            fp_cat.astype(np.float32),
            fp_reag.astype(np.float32),
        ]
        if use_desc:
            desc_delta = (sum_descriptors(products) - sum_descriptors(reactants)).astype(np.float32)
            feat_parts.append(desc_delta)
        feat_parts.append(np.array([cat_missing, reag_missing], dtype=np.float32))
        feat = np.concatenate(feat_parts)

        # Note: see the surrounding code for details.
        X_blocks.append(feat)
        y_list.append(np.clip(y, 0.0, 100.0))

        # Note: see the surrounding code for details.
        if (row_idx + 1) % 10000 == 0:
            X_part = np.vstack(X_blocks) if X_blocks else np.zeros((0, base_dim), dtype=np.float32)
            y_part = np.array(y_list, dtype=np.float32)
            np.save(x_part_path, X_part)
            np.save(y_part_path, y_part)
            with open(done_idx_path, 'w') as f:
                f.write(str(row_idx))
            print(f'Saved resume point: {row_idx}')

    # Note: see the surrounding code for details.
    if X_blocks:
        X_final = np.vstack(X_blocks)
        y_final = np.array(y_list, dtype=np.float32)
        np.save(final_x_path, X_final)
        np.save(final_y_path, y_final)
        # Note: see the surrounding code for details.
        for f in [x_part_path, y_part_path, done_idx_path]:
            if os.path.exists(f):
                os.remove(f)
        print(f'Feature extraction status: {split_name}')
        print(f"   - X: {final_x_path}")
        print(f"   - y: {final_y_path}")
        return X_final, y_final
    else:
        raise ValueError(f'Feature extraction status: {split_name}')

# Note: see the surrounding code for details.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Training CSV path')
    parser.add_argument('--split_name', required=True, help='Message')
    parser.add_argument('--save-dir', required=True, help='Histogram output path')
    parser.add_argument('--bits', type=int, default=2048, help='Morgan fingerprint length')
    parser.add_argument('--radius', type=int, default=2, help='Morgan radius')
    parser.add_argument('--use-desc', action='store_true', help='Enable molecular descriptors')
    args = parser.parse_args()

    # Note: see the surrounding code for details.
    os.makedirs(args.save_dir, exist_ok=True)

    # Note: see the surrounding code for details.
    required_cols = {'reaction_smiles', 'transition_metal_catalyst', 'other_reagent', 'yield'}
    df = pd.read_csv(args.csv, sep=None, engine='python')
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    # Note: see the surrounding code for details.
    build_features_with_resume(
        df=df,
        nBits=args.bits,
        radius=args.radius,
        save_dir=args.save_dir,
        split_name=args.split_name,
        use_desc=args.use_desc
    )

if __name__ == '__main__':
    main()
