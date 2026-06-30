import pandas as pd
import re

# Note: see the surrounding code for details.
df = pd.read_csv("/sharefs/lijl/xingjie/dataset/reagents_split_by_catalyst.csv")

# Note: see the surrounding code for details.
pattern = re.compile(r'\[Li\]\s*\[AlH4\]')          # Note: see the surrounding code for details.

def fix_li_alh4(cell: str | float):
    if not isinstance(cell, str):
        return cell                                  # Note: see the surrounding code for details.
    return pattern.sub('[Li+][AlH4-]', cell)        # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
df["other_reagent"] = df["other_reagent"].apply(fix_li_alh4)

# Note: see the surrounding code for details.
df.to_csv("/sharefs/lijl/xingjie/dataset/reagents_split_by_catalyst_fixed.csv", index=False)

print('Status message.')
print(df["other_reagent"].head())
