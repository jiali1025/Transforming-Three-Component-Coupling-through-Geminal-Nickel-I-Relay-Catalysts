# split_dataset.py
# -----------------------------------------------------------
# Note: see the surrounding code for details.
# -----------------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np

# Note: see the surrounding code for details.
PROJECT_ROOT   = Path(__file__).resolve().parent
CSV_PATH       = PROJECT_ROOT / "data" / "final_data_featurizable_max.csv"
OUT_DIR        = PROJECT_ROOT / "splits"          # output directory
VAL_SIZE       = 10_000
TEST_SIZE      = 10_000
TM_RATIO       = 0.30                      # Note: see the surrounding code for details.
TM_COL         = "transition_metal_catalyst"                # Note: see the surrounding code for details.
RAND_SEED      = 42                        # Note: see the surrounding code for details.
# -------------------------------------------


def sample_exact_ratio(df_tm: pd.DataFrame,
                       df_no: pd.DataFrame,
                       total: int,
                       ratio: float,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Sample from TM and non-TM subsets to match the requested ratio exactly."""
    need_tm  = int(round(total * ratio))
    need_no  = total - need_tm

    if len(df_tm) < need_tm or len(df_no) < need_no:
        raise ValueError(
            f'Status: {need_tm} {need_no}'
            f'Status: {len(df_tm)} {len(df_no)}')

    idx_tm  = rng.choice(df_tm.index,  size=need_tm, replace=False)
    idx_no  = rng.choice(df_no.index,  size=need_no, replace=False)
    return pd.concat([df_tm.loc[idx_tm], df_no.loc[idx_no]])


def split_dataset(df: pd.DataFrame):
    rng = np.random.default_rng(RAND_SEED)

    # Note: see the surrounding code for details.
    df["__tm__"] = df[TM_COL].notna().astype(int)

    df_tm    = df[df["__tm__"] == 1]
    df_no_tm = df[df["__tm__"] == 0]

    # Note: see the surrounding code for details.
    val_df = sample_exact_ratio(df_tm, df_no_tm, VAL_SIZE, TM_RATIO, rng)

    # Note: see the surrounding code for details.
    remain_tm    = df_tm.drop(val_df.index, errors="ignore")
    remain_no_tm = df_no_tm.drop(val_df.index, errors="ignore")

    # Note: see the surrounding code for details.
    test_df = sample_exact_ratio(remain_tm, remain_no_tm, TEST_SIZE, TM_RATIO, rng)

    # Note: see the surrounding code for details.
    used_idx  = val_df.index.union(test_df.index)
    train_df  = df.drop(index=used_idx)

    # Note: see the surrounding code for details.
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        ratio = part["__tm__"].mean()
        print(f"{name:<5s} | rows={len(part):>8,} | TM_ratio={ratio:5.1%}")

    # Note: see the surrounding code for details.
    for part in (train_df, val_df, test_df):
        part.drop(columns="__tm__", inplace=True)

    return train_df, val_df, test_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Path: {CSV_PATH}')
    df = pd.read_csv(CSV_PATH)

    train_df, val_df, test_df = split_dataset(df)

    # Note: see the surrounding code for details.
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv",   index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    print(f'Path: {OUT_DIR.resolve()}')


if __name__ == "__main__":
    main()

'''
train | rows=8,421,390 | TM_ratio= 8.4%
val   | rows=  10,000 | TM_ratio=30.0%
test  | rows=  10,000 | TM_ratio=30.0%
'''
