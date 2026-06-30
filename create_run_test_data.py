import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Note: see the surrounding code for details.
df = pd.read_csv(DATA_DIR / "reagents_split_by_catalyst.csv")

# Note: see the surrounding code for details.
df_sample = df.sample(n=1000, random_state=42)

# Note: see the surrounding code for details.
DATA_DIR.mkdir(parents=True, exist_ok=True)
df_sample.to_csv(DATA_DIR / "reagents_split_sample_1000.csv", index=False)

print(f"Sampling complete; saved to: {DATA_DIR / 'reagents_split_sample_1000.csv'}")
