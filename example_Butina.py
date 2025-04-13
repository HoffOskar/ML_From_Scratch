### Imports
from utils.chem import Butina
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### Data
df = pd.read_pickle("Clustering/data/morgan_2048_df.pkl")

### Fit the model
model = Butina().fit(df)

### Clustering
pred_df = model.predict(threshold=0.66)

### Countplot
plt.figure(figsize=(10, 5))
ax = sns.countplot(x="Cluster", data=pred_df)

### Axis ticks
ticks = ax.get_xticks()
ax.set_xticks(ticks[::100])  # Show every 10th cluster
ax.set_xticklabels([int(t) for t in ticks[::100]], rotation=0)

### Set the title and labels
plt.title("Number of compounds in each cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.show()