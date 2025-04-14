import numpy as np
import pandas as pd
from tqdm import tqdm


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

        Raises
        ------
        ValueError
            If the input DataFrame index is not unique.
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
        self.cluster_df = self.compound_df["Cluster"].value_counts().sort_index()

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


class KennardStone:
    """
    Kennard-Stone algorithm for data splitting.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame with the input data with binary values (e.g. Morgan fingerprints). Requires the index to be unique.

    """

    def __init__(self):
        self.df = None
        self.distance_matrix = None
        self.subset_mask = None

    def fit(self, df):
        """
        Calculate the distance matrix.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with binary values (0/1).

        Returns
        -------
        self : KennardStone
            The fitted instance of the KennardStone class.

        Side Effects
        ------------
        - Updates the distance_matrix attribute.

        Raises
        ------
        ValueError
            If the input DataFrame is not binary (0/1).
        ValueError
            If the input DataFrame index is not unique.
        """
        ### Check if the input DataFrame has unique indexes
        if not df.index.is_unique:
            raise ValueError("Input DataFrame indexes must be unique.")

        ### Save the input DataFrame as attribute
        self.df = df

        ### Convert the DataFrame to a numpy array
        X = df.to_numpy(dtype=np.int32)

        ### Check if the input is binary
        if not np.isin(X, [0, 1]).all():
            raise ValueError("Input must be binary (0/1) for Hamming distance")

        ### Calculate the intersection matrix
        intersection = X @ X.T

        ### Count the number of 1s for each compound
        X_norm = X.sum(axis=1).reshape(-1, 1)

        ### Calculate the Hamming distance matrix
        dist_matrix = X_norm + X_norm.T - 2 * intersection

        ### Euclidean distance matrix if needed
        # dist_matrix = np.sqrt(dist_matrix)

        ### Save the input data and distance matrix as attribute
        self.df = df
        self.distance_matrix = pd.DataFrame(
            dist_matrix, index=df.index, columns=df.index
        )

        return self

    def split(self, subset_size=0.1, warm_start=False, warm_subset=None):
        """
        Define subset based on the Kennard-Stone algorithm.

        Parameters
        ----------
        subset_size : float
            The size of the subset as a fraction of the total dataset size. Default is 0.1.

        Returns
        -------
        subset : pd.DataFrame
            Subset with selected compounds. Can be used as representative diversity or training set.

        Side Effects
        ------------
        - Updates the subset_mask attribute.

        Raises
        ------
        ValueError
            If the distance matrix is not calculated.
        """
        ### Check if the distance matrix is calculated
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not calculated. Call fit() first.")

        ### Calculate the number of samples in the test set
        n_total_samples = len(self.df)
        n_sub_samples = int(n_total_samples * subset_size)

        ### Initialize the boolean mask for the subset
        subset_mask = pd.Series(False, index=self.df.index)

        ### Define the first sample (.idxmax() only returns the first occurrence)
        subset_mask.loc[self.distance_matrix.max().idxmax()] = True

        ### Sequential Kennard-Stone algorithm
        while subset_mask.sum() < n_sub_samples:
            # 1 Subset the full distance matrix: rows: unassigned samples, columns: assigned samples
            # 2 Find the nearest assigned neighbor for each unassigned sample
            # 3 Get the index of the unassigned sample with the maximum distance to its nearest assigned neighbor
            # 4 Assign this unassigned sample to the subset
            # 5 Repeat until the subset size is reached
            subset_mask.loc[
                self.distance_matrix[subset_mask].T[~subset_mask].min(axis=1).idxmax()
            ] = True

        ### Assign the subset mask as class attribute
        self.subset_mask = subset_mask

        ### Return the subset and the remaining samples
        return self.df[subset_mask].copy()
