#!/usr/bin/env python
import argparse, sys
import pandas as pd
import numpy as np

# Note: see the surrounding code for details.
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

CANDIDATES = ["yield", "Yield", "Y", "y", "target", "label", "labels"]

def autodetect_label(df: pd.DataFrame):
    # Note: see the surrounding code for details.
    for c in CANDIDATES:
        if c in df.columns:
            return c
    # Note: see the surrounding code for details.
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if num_cols:
        return num_cols[-1]
    raise ValueError('No numeric label column found. Specify one with --label.')

def freedman_diaconis_bins(x: np.ndarray):
    # Note: see the surrounding code for details.
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 10
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = max(q75 - q25, 1e-9)
    h = 2 * iqr * (len(x) ** (-1/3))
    if h <= 0:
        return 10
    bins = int(np.ceil((x.max() - x.min()) / h))
    return max(10, min(bins, 100))

def main():
    ap = argparse.ArgumentParser(description='Inspect the y distribution in the train split')
    ap.add_argument("--csv", required=True, help='Training CSV path')
    ap.add_argument("--label", default=None, help='Label column name (auto-detected if omitted)')
    ap.add_argument("--clip01", action="store_true",
                    help='Clip y values to [0, 100].')
    ap.add_argument("--save", default="train_y_hist.png",
                    help='Histogram output path.')
    ap.add_argument("--no-plot", action="store_true", help='Print statistics only; do not plot')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    col = args.label or autodetect_label(df)

    if col not in df.columns:
        print(f'Specified column is not in the CSV; available columns: {col} {list(df.columns)}', file=sys.stderr)
        sys.exit(1)

    y = df[col].astype(float).to_numpy()
    y = y[~np.isnan(y)]

    if args.clip01:
        y = np.clip(y, 0, 100)

    n = len(y)
    if n == 0:
        print('[ERROR] y is empty.')
        sys.exit(1)

    # Note: see the surrounding code for details.
    stats = {
        "count": n,
        "min": float(np.min(y)),
        "p10": float(np.percentile(y, 10)),
        "p25": float(np.percentile(y, 25)),
        "median": float(np.median(y)),
        "p75": float(np.percentile(y, 75)),
        "p90": float(np.percentile(y, 90)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=0)),
        "var": float(np.var(y, ddof=0)),
        "pct==0": float((y == 0).mean() * 100),
        "pct>=90": float((y >= 90).mean() * 100),
        "pct<=10": float((y <= 10).mean() * 100),
    }

    print(f'y column: {col}')
    for k, v in stats.items():
        print(f"  {k:>7}: {v:.4f}" if isinstance(v, float) else f"  {k:>7}: {v}")

    # Note: see the surrounding code for details.
    bins = freedman_diaconis_bins(y)
    hist, edges = np.histogram(y, bins=bins)
    print('Histogram bins (text counts):')
    for i in range(len(hist)):
        left, right, cnt = edges[i], edges[i+1], int(hist[i])
        print(f"  [{left:6.2f}, {right:6.2f}) : {cnt}")

    # Note: see the surrounding code for details.
    if not args.no_plot and HAS_MPL:
        plt.figure()
        plt.hist(y, bins=bins, edgecolor="black")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title("Train y distribution")
        plt.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f'Histogram saved: {args.save}')
    elif not HAS_MPL and not args.no_plot:
        print('matplotlib is not installed; skipped plotting.')

if __name__ == "__main__":
    main()
