### Imports
import numpy as np
import pandas as pd


### Node class

class Node:
    def __init__(self, feature_idx=None, x_threshold=None, left_child=None, right_child=None, node_value=None):
        self.feature_idx = feature_idx
        self.x_threshold = x_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.node_value = node_value



### RegressionTree class

class RegressionTree:
    
    def __init__(self, max_depth=None, min_samples_split=2, root=None):
        self.max_depth = max_depth                      # number of sequential splits allowed
        self.min_samples_split = min_samples_split      # minimum number of samples required to split an internal node
        self.root = None                                # root node of the decision tree
 
    
    def fit(self, X_train, y_train):
        sample_idx = np.arange(X_train.shape[0])
        self.root = self._build_tree(X_train, y_train, sample_idx, depth=0)


    def predict(self, X_test):
        return np.array([self._predict(x, self.root) for x in X_test])


    ### Helper functions


    def _predict(self, x, node):
        '''
        Recursively predict the target value for a single sample.
        '''
        
        ### Return the node value if the node is a leaf
        if node.left_child is None:
            return node.node_value
        
        ### Recursively call the function on the left or right child
        if x[node.feature_idx] < node.x_threshold:
            return self._predict(x, node.left_child)
        else:
            return self._predict(x, node.right_child)

    
    def _build_tree(self, X_train, y_train, sample_idx, depth):
        '''
        Recursively build the decision tree.
        '''
        
        ### Create a new node
        node = Node()
        
        ### Stop splitting if max_depth is reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(node_value=y_train[sample_idx].mean())
            #return None
        
        ### Stop splitting if node has too few samples
        if len(sample_idx) < self.min_samples_split:
            return Node(node_value=y_train[sample_idx].mean())
            #return None
        
        ### Find the best split
        split = self._find_best_split(X_train[sample_idx], y_train[sample_idx])
        
        ### Stop splitting if no split is found
        if split is None:
            return Node(node_value=y_train[sample_idx].mean())
            #return None
        
        ### Unpack the split
        feature_idx, x_threshold, left_idx, right_idx = split
        
        ### Create node parameters and initiate left and right child nodes
        node.feature_idx = feature_idx
        node.x_threshold = x_threshold
        node.node_value = y_train[sample_idx].mean()

        node.left_child = self._build_tree(X_train, y_train, sample_idx[left_idx], depth+1)
        node.right_child = self._build_tree(X_train, y_train, sample_idx[right_idx], depth+1)
        
        return node

    
    def _split_feature(self, x_train, y_train):
        '''
        Split a feature vector (1D) into two parts with minimal MSE.

        Parameters:
            x_train (np.array): 1D array of feature values
            y_train (np.array): 1D array of target values

        Returns:
            min_MSE: The MSE of the best split
            x_threshold: The threshold value for the best split
            left_idx: The indices of the left data
            right_idx: The indices of the right data
        '''
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
            left_count = i
            right_count = n - i
            
            left_sum_y = cumsum_y[i-1]
            right_sum_y = total_sum_y - left_sum_y
            
            left_sum_y_sq = cumsum_y_sq[i-1]
            right_sum_y_sq = total_sum_y_sq - left_sum_y_sq
            
            left_mse = left_sum_y_sq - (left_sum_y**2) / left_count
            right_mse = right_sum_y_sq - (right_sum_y**2) / right_count
            
            mse_values[i-1] = left_mse + right_mse

        ### Find the best split point
        split_idx = np.argmin(mse_values)
        x_threshold = x_train[sorted_indices[split_idx]]

        ### Return the MSE of the best split point and the indices of the left and right data
        return np.min(mse_values), x_threshold, sorted_indices[:split_idx + 1], sorted_indices[split_idx + 1:]
    

    def _find_best_split(self, X_train, y_train):
        '''
        Determine the best split for a node.
        '''
        
        splits_df = pd.DataFrame([self._split_feature(X_train[:, col], y_train) for col in range(X_train.shape[1])], 
                          columns=['MSE', 'x_threshold', 'left_idx', 'right_idx'])
        feature_idx = splits_df['MSE'].idxmin()
        left_idx = splits_df.loc[feature_idx, 'left_idx']
        right_idx = splits_df.loc[feature_idx, 'right_idx']
        x_threshold = splits_df.loc[feature_idx, 'x_threshold']

        return feature_idx, x_threshold, left_idx, right_idx