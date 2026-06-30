# map_local2global.py
"""Remove rows listed in bad_rows_global_after_fix.txt and write a cleaned CSV while preserving other columns."""

from pathlib import Path

FULL_IDX_FILE  = "bad_row_indices.txt"      # Note: see the surrounding code for details.
LOCAL_BAD_FILE = "bad_rows_after_fix.txt"   # Note: see the surrounding code for details.
OUT_FILE       = "bad_rows_global_after_fix.txt"

# Note: see the surrounding code for details.
full_idx  = [int(x) for x in Path(FULL_IDX_FILE).read_text().splitlines() if x.strip()]
local_bad = [int(x) for x in Path(LOCAL_BAD_FILE).read_text().splitlines() if x.strip()]

# Note: see the surrounding code for details.
global_bad = [full_idx[i] for i in local_bad]


# Note: see the surrounding code for details.
Path(OUT_FILE).write_text("\n".join(map(str, global_bad)))
print(f"mapped {len(local_bad):,} local -> {len(global_bad):,} global indices")
print("output  ->", Path(OUT_FILE).resolve())
