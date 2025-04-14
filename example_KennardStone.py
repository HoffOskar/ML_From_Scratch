### Imports
import pandas as pd

### Utils
from utils.chem import KennardStone

### Data
df = pd.read_pickle("Clustering/data/morgan_2048_df.pkl")

### Instantiate and fit the model
model = KennardStone().fit(df)

### Split the data
diversity_set = model.split(subset_size=0.10)


print(diversity_set.head())
