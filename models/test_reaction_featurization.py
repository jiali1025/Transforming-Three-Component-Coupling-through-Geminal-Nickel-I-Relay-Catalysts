import pandas as pd
from pathlib import Path
from tqdm import tqdm
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer
from chemprop.data.datapoints import ReactionDatapoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Note: see the surrounding code for details.
df = pd.read_csv(PROJECT_ROOT / "data" / "reagents_split_sample_1000.csv")

featurizer = CondensedGraphOfReactionFeaturizer()
mol_graphs = []

# Note: see the surrounding code for details.
for i, smi in tqdm(enumerate(df["reaction_smiles"]), total=len(df)):
    try:
        dp = ReactionDatapoint.from_smi(smi)
        mol_graph = featurizer((dp.rct, dp.pdt))
        mol_graphs.append(mol_graph)

        # Note: see the surrounding code for details.
        # print(dp.rct, dp.pdt)
    except Exception as e:
        print(f"ERROR: Error at row {i}\nSMILES: {smi}\nReason: {e}")
        mol_graphs.append(None)

print(" Done! Total featurized:", sum(g is not None for g in mol_graphs))
