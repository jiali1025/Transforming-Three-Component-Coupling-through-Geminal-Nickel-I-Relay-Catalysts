import pandas as pd

# Note: see the surrounding code for details.
SRC = "/sharefs/lijl/xingjie/dataset/final_data_featurizable_max.csv"

# Note: see the surrounding code for details.
df = pd.read_csv(SRC, nrows=5)

# Note: see the surrounding code for details.
print('Status message.')
print(df.columns.tolist())