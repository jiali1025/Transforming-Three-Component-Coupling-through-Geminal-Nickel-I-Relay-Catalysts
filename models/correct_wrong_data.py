# fix_reagents_inplace.py
"""Return the maximum metal-atom count among reagent molecules split by '|'."""

import re, pandas as pd, os, shutil, tempfile
from pathlib import Path
from tqdm import tqdm

SRC  = "/sharefs/lijl/xingjie/dataset/reagents_split_by_catalyst.csv"
DST  = "/sharefs/lijl/xingjie/dataset/reagents_split_by_catalyst_fixed.csv"
CHUNK = 200_000                       # Note: see the surrounding code for details.

# Note: see the surrounding code for details.
FIX_MAP = {
    # Note: see the surrounding code for details.
    r'\[Li\]\s*\[AlH4\]' : '[Li+].[AlH4-]',
    r'\[Na\]\s*\[AlH4\]' : '[Na+].[AlH4-]',
    r'\[Na\]\s*\[BH4\]'  : '[Na+].[BH4-]',
    r'\[K\]\s*\[BH4\]'   : '[K+].[BH4-]',
    # Note: see the surrounding code for details.
    r'B\(Br\)\(Br\)\(Br\)Br' : '[B-](Br)(Br)(Br)Br',
    # Note: see the surrounding code for details.
    r'Br\[BrH\]Br'       : '[Br+].[Br-].Br',
    r'Br\[Br-]Br'        : '[Br+].[Br-].Br',
    r'Br\[Br]\(Br\)Br'   : '[Br+].[Br-].Br',
    # Note: see the surrounding code for details.
    r'\[Li\]\s*\[Cl\]'   : '[Li+].[Cl-]',
    r'\[Cs\]F'           : '[Cs+].[F-]',
    r'\[Cs\]Cl'          : '[Cs+].[Cl-]',
    # Note: see the surrounding code for details.
    r'\[BH3\]\[NH3\]'            : '[BH4-].[NH3+]',
    r'\[BH3\]\[NH\]\(C\)C'       : '[BH3-].[NH+(C)C]',
    r'\[BH3\]\[NH\]1CCOCC1'      : '[BH3-].[NH+1CCOCC1]',
    r'\[BH3\]\[N]\(C\)\(C\)C'    : '[BH3-].[N+(C)(C)C]',
    # Note: see the surrounding code for details.
    r'\[N]\(=O\)=O'   : '[N+](=O)[O-]',
    r'\[O-\]=O'       : '[O-].[O-]',
}
PAT_SUB = [(re.compile(p), r) for p, r in FIX_MAP.items()]

def fix_cell(x: str | float):
    if not isinstance(x, str) or not x.strip():
        return x
    s = x
    for pat, repl in PAT_SUB:
        s = pat.sub(repl, s)
    return s

# Note: see the surrounding code for details.
cols_to_fix = ("transition_metal_catalyst", "other_reagent")
tmpfile = Path(tempfile.mkstemp(suffix=".csv", dir=Path(DST).parent)[1])
first   = True

total = sum(1 for _ in open(SRC, 'rb')) - 1     # Note: see the surrounding code for details.
with tqdm(total=total, unit="lines") as bar:
    for chunk in pd.read_csv(SRC, chunksize=CHUNK):
        for col in cols_to_fix:
            if col in chunk.columns:
                chunk[col] = chunk[col].apply(fix_cell)

        mode = "w" if first else "a"
        header = first
        chunk.to_csv(tmpfile, mode=mode, header=header, index=False)
        first = False

        bar.update(len(chunk))

# Note: see the surrounding code for details.
shutil.move(tmpfile, DST)
print(f'Output file: {DST}')
