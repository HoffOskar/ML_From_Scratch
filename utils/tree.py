### Imports
import numpy as np
import pandas as pd


### Node class

class Node:
    '''
    A class representing a single node in the decision tree.

    Attributes:
        feature_idx (int): The positional index of the feature used for splitting at this node.
        x_threshold (float): The threshold value used for splitting the feature.
        left_child (Node): The left subtree (where feature values are < threshold).
        right_child (Node): The right subtree (where feature values are >= threshold).
        node_value (float): The predicted value if the node is a leaf.
    '''
    def __init__(self, feature_idx=None, x_threshold=None, left_child=None, right_child=None, node_value=None):
        self.feature_idx = feature_idx
        self.x_threshold = x_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.node_value = node_value



### RegressionTree class

class RegressionTree:
    """
    A custom implementation of a regression decision tree.

    Attributes:
        max_depth (int): The number of sequential splits allowed. Default is None (unlimited depth).
        min_samples_split (int): The minimum number of samples required to split a node. Default is 2.
        root (Node): The root node of the tree.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, root=None):
        self.max_depth = max_depth                      
        self.min_samples_split = min_samples_split      
        self.root = root                                
 
    
    def fit(self, X_train, y_train):
        """
        Train the regression tree by recursively splitting the data.

        Parameters:
            X_train (np.array): The feature matrix of shape (n_samples, n_features).
            y_train (np.array): The target values of shape (n_samples,).
        
        Side effects:
            - Updates self.root with the trained decision tree.
        """
        sample_idx = np.arange(X_train.shape[0])
        self.root = self._build_tree(X_train, y_train, sample_idx, depth=0)


    def predict(self, X_test):
        """
        Predict the target values for new data points.

        Parameters:
            X_test (np.array): The feature matrix of shape (n_samples, n_features).
        
        Returns:
            An np.array containing predicted values for each sample.
        """
        return np.array([self._predict(x, self.root) for x in X_test])


    ### Helper functions for RegressionTree.fit()

    
    def _build_tree(self, X_train, y_train, sample_idx, depth):
        '''
        Recursively build the decision tree.

        Parameters:
            X_train (np.array): The feature matrix of shape (n_samples, n_features).
            y_train (np.array): The target values of shape (n_samples,).
            sample_idx (np.array): Indices of samples assigned to the current node.
            depth (int): The current depth of the tree.

        Returns:
            node (Node): The root node of the created subtree or a leaf node if stopping criteria are met.

        Stopping criteria:
            The maximum depth (`max_depth`) is reached.
            The number of samples is below `min_samples_split`.
            No valid split is found.

        Side effects:
            The node's attributes feature_idx, x_threshold, and node_value are defined. 
            Recursively creates node's left and right child.
        '''
        
        ### Create a new node
        node = Node()
        
        ### Stop splitting if max_depth is reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(node_value=y_train[sample_idx].mean())
        
        ### Stop splitting if node has too few samples
        if len(sample_idx) < self.min_samples_split:
            return Node(node_value=y_train[sample_idx].mean())
        
        ### Find the best split
        split = self._find_best_split(X_train[sample_idx], y_train[sample_idx])
        
        ### Stop splitting if no split is found
        if split is None:
            return Node(node_value=y_train[sample_idx].mean())
        
        ### Unpack the split
        feature_idx, x_threshold, left_idx, right_idx = split
        
        ### Update the node attributes
        node.feature_idx = feature_idx
        node.x_threshold = x_threshold
        node.node_value = y_train[sample_idx].mean()

        ### Initiate left and right child nodes by recursively calling the function
        node.left_child = self._build_tree(X_train, y_train, sample_idx[left_idx], depth+1)
        node.right_child = self._build_tree(X_train, y_train, sample_idx[right_idx], depth+1)
        
        return node

    
    def _find_best_split(self, X_train, y_train):
        '''
        Determine the best split for a node.

        Parameters:
            X_train (np.array): The feature matrix of shape (n_samples, n_features).
            y_train (np.array): The target values of shape (n_samples,).

        Returns:
            feature_idx (int): The index of the best feature for splitting.
            x_threshold (float): The best threshold value for the split.
            left_idx (np.array): Indices of samples in the left subset.
            right_idx (np.array): Indices of samples in the right subset.
        '''
        
        ### Store split results in a DataFrame
        splits_df = pd.DataFrame([self._split_feature(X_train[:, col], y_train) for col in range(X_train.shape[1])], 
                          columns=['MSE', 'x_threshold', 'left_idx', 'right_idx'])
        
        ### Return None if no valid split is found
        if splits_df['MSE'].min() == np.inf:
            return None
        
        ### Find the best split
        feature_idx = splits_df['MSE'].idxmin()
        left_idx = splits_df.loc[feature_idx, 'left_idx']
        right_idx = splits_df.loc[feature_idx, 'right_idx']
        x_threshold = splits_df.loc[feature_idx, 'x_threshold']

        return feature_idx, x_threshold, left_idx, right_idx


    def _split_feature(self, x_train, y_train):
        '''
        Split a feature vector (1D) into two parts with minimal MSE.

        Parameters:
            x_train (np.array): 1D array of feature values
            y_train (np.array): 1D array of target values

        Returns:
            min_MSE (float): The MSE of the best split
            x_threshold (float): The threshold value for the best split
            left_idx (int): The indices of the left data
            right_idx (int): The indices of the right data
        '''
        
        ### Return None if all feature values are the same
        if len(np.unique(x_train)) == 1:  
            return np.inf, None, None, None
        
        ### Return None if there are not enough points to split
        if len(x_train) < 2:
            return np.inf, None, None, None
        
        ### Sort the target by feature values
        sorted_indices = np.argsort(x_train)
        y_train_sorted = y_train[sorted_indices]

        ### Compute cumulative sums
        cumsum_y = np.cumsum(y_train_sorted)            # cumulative sum of target values
        cumsum_y_sq = np.cumsum(y_train_sorted**2)      # cumulative sum of squared target values

        ### Total sum and sum of squares
        total_sum_y = cumsum_y[-1]
        total_sum_y_sq = cumsum_y_sq[-1]

        ### Number of samples
        n = len(y_train_sorted)

        ### Initialize array to store MSE values
        mse_values = np.zeros(n-1)

        ### Calculate MSE for each split point
        for i in range(1, n):
            
            ### Index
            left_count = i
            right_count = n - i
            
            ### Calculate cummulative sums by slicing
            left_sum_y = cumsum_y[i-1]
            right_sum_y = total_sum_y - left_sum_y
            
            ### Calculate cummulative sums of squares by slicing
            left_sum_y_sq = cumsum_y_sq[i-1]
            right_sum_y_sq = total_sum_y_sq - left_sum_y_sq

            ### Calculate MSE
            left_mse = left_sum_y_sq - (left_sum_y**2) / left_count
            right_mse = right_sum_y_sq - (right_sum_y**2) / right_count
            
            mse_values[i-1] = left_mse + right_mse

        ### Find the best split point
        split_idx = np.argmin(mse_values)
        x_threshold = x_train[sorted_indices[split_idx]]

        ### Return the MSE of the best split point and the indices of the left and right data
        return np.min(mse_values), x_threshold, sorted_indices[:split_idx + 1], sorted_indices[split_idx + 1:]
    

   ### Helper functions for RegressionTree.predict()


    def _predict(self, x, node):
        '''
        Recursively predict the target value for a single sample.

        Parameters:
            x (np.array): A 1D array representing a single sample.
            node (Node): The current node in the decision tree.

        Returns:
            node_value (float): The predicted value from the leaf node.
        '''
        
        ### Return the node value if the node is a leaf
        if node.left_child is None:
            return node.node_value
        
        ### Recursively call the function on the left or right child
        elif x[node.feature_idx] < node.x_threshold:
            return self._predict(x, node.left_child)
        else:
            return self._predict(x, node.right_child)