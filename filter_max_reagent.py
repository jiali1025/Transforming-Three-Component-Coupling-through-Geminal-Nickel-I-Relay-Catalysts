import pandas as pd

SRC = "/sharefs/lijl/xingjie/dataset/pistachio_ionic_merged.csv"
DST = "/sharefs/lijl/xingjie/dataset/pistachio_filter_ionic.csv"

# Note: see the surrounding code for details.
df = pd.read_csv(SRC)   # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
df["n_reagents"] = df["reagents"].str.split('.').str.len().fillna(0).astype('uint8')

# Note: see the surrounding code for details.
df_clean = df[df["n_reagents"] <= 6].reset_index(drop=True)

print(f'Rows retained after filtering: {len(df_clean):,} {len(df):,} {len(df_clean)/len(df):.2%}')

df_clean.to_csv(DST, index=False)
print('Status message.', DST)