import numpy as np
import pandas as pd
from tqdm import tqdm

class KennardStone:
    def __init__(self, X):
        self.X = X
        

class Butina:
    """
    Butina clustering algorithm for fingerprints.

    Attributes
    ----------
    threshold : float
        Similarity threshold for clustering.
    df : pd.DataFrame
        Input DataFrame with the fingerprints.
    similarity_df : pd.DataFrame
        DataFrame with the similarity matrix.
    compound_df : pd.DataFrame
        DataFrame with the cluster assignments. Same length and index as df.
    cluster_df : pd.DataFrame
        DataFrame with the cluster summary. 
    """

    def __init__(self):
        self.threshold = None
        self.df = None
        self.similarity_df = None
        self.cluster_df = None
        self.compound_df = None

    def fit(self, df):
        """
        Compute the similarity matrix.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the fingerprints. Requires the index to be unique.
        
        Returns
        -------
        self : Butina
            The instance of the Butina class.

        Side Effects
        ------------
        - Updates the similarity_df attribute with the similarity matrix.
        """

        ### Check if the input DataFrame has unique indexes
        if not df.index.is_unique:
            raise ValueError("Input DataFrame indexes must be unique.")

        ### Assigne the input DataFrame to the instance
        self.df = df.copy()

        ### Convert the DataFrame with fingerprints to a NumPy array
        X = df.to_numpy().astype(int)

        ### Compute the intersection of the fingerprints
        intersect = X @ X.T

        ### Compute the number of on bits in each fingerprint
        on_bits = X.sum(axis=1)

        ### Compute the denominator
        denom = on_bits[:, None] + on_bits[None, :] - intersect

        ### Compute the distance matrix
        sim = np.divide(
            intersect,
            denom,
            out=np.zeros_like(intersect, dtype=float),
            where=denom != 0,
        )

        ### Set the diagonal to 0
        np.fill_diagonal(sim, 0)

        ### Set the similarity matrix
        self.similarity_df = pd.DataFrame(
            sim, index=self.df.index, columns=self.df.index
        )
        return self

    def predict(self, threshold=0.75):
        """
        Perform the clustering using the Butina algorithm.

        Parameters
        ----------
        threshold : float
            Similarity threshold for clustering. Default is 0.75.

        Returns
        -------
        compound_df : pd.DataFrame
            DataFrame with the cluster assignments. Same length and index as df.
        cluster_df : pd.DataFrame
            DataFrame with the cluster summary. 

        Side Effects
        ------------
        - Updates the compound_df attribute with the cluster assignments.
        """

        ### Instantiate the DataFrame to store the cluster assignments with NaN values
        self.compound_df = pd.DataFrame(index=self.df.index, columns=["Cluster"])
        self.compound_df["Centroid"] = False
        self.compound_df["Singleton"] = False

        ### Set the threshold
        self.threshold = threshold

        ### Verbose output
        print("Clustering - threshold: ", threshold)

        ### Progress bar
        with tqdm(total=100) as pbar:
            ### Loop until all compounds are assigned to a cluster
            while self.compound_df["Cluster"].isna().any():
                ### Get the next cluster members
                cluster_members = self._assign_next_cluster()

                ### Update the progress bar
                pbar.update(cluster_members / len(self.df) * 100)

        ### Format data types
        self.compound_df["Cluster"] = self.compound_df["Cluster"].astype(int)

        ### Cluster summary
        self.cluster_df = self.compound_df['Cluster'].value_counts().sort_index()

        return self.compound_df

    def _assign_next_cluster(self):
        """
        Define the next cluster by identifying the centroid (compound with most neighbors) and its neighbors.

        Returns
        -------
        int
            The number of assigned cluster members.

        Side Effects
        ------------
        - Updates the compound_df DataFrame with the cluster assignments.
        """

        ### Define the ID of the next cluster (starting from 0)
        last_cluster_id = self.compound_df["Cluster"].max()
        next_cluster_id = 0 if pd.isna(last_cluster_id) else int(last_cluster_id + 1)

        ### Subset the similarity matrix for all compounds without cluster assignment
        unassigned_mask = self.compound_df["Cluster"].isna()
        free_sim_df = self.similarity_df.loc[unassigned_mask, unassigned_mask]

        ### Identify the centroid of the next cluster (the compound with most neighbors)
        centroid_id = (free_sim_df > self.threshold).sum(axis=1).idxmax()

        ### Identify the cluster members (neighbors of the centroid)
        neigb_id = free_sim_df.loc[free_sim_df[centroid_id] > self.threshold].index

        ### Assign the cluster ID to the centroid and its neighbors
        self.compound_df.loc[centroid_id, "Cluster"] = next_cluster_id
        self.compound_df.loc[neigb_id, "Cluster"] = next_cluster_id

        ### Assign the Centroid flag
        self.compound_df.loc[centroid_id, "Centroid"] = True

        ### Assign the Singleton flag to the centroid if it has no neighbors
        if len(neigb_id) == 0:
            self.compound_df.loc[centroid_id, "Singleton"] = True

        ### Return the number of assigned cluster members
        return len(neigb_id) + 1
