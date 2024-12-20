import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth: int=5):
        self.max_depth = max_depth
    
    def best_split(self, X, y):
        best_gini, best_feature, best_value = float('inf'), None, None
        n = X.shape[1]

        for feature in range(n):
            feature_vals = np.unique(X[:, feature])
            for val in feature_vals:
                # Split indices
                left_ind = X[:, feature] < val
                right_ind = ~left_ind

                # label check
                if len(y[left_ind]) == 0 or len(y[right_ind]) == 0: continue

                # Calculate Gini impurity
                gini = self.Gini(y[left_ind], y[right_ind])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_value = val

        return best_feature, best_value

    def build_tree(self, X, y, depth: int):
        best_feature, best_value = self.best_split(X, y)
        if depth == self.max_depth or best_feature is None:
            return np.bincount(y).argmax()  # Return the most common class label

        # Split indices
        left_ind = X[:, best_feature] < best_value
        right_ind = ~left_ind

        # Recursively build subtrees
        left_subtree = self.build_tree(X[left_ind], y[left_ind], depth + 1)
        right_subtree = self.build_tree(X[right_ind], y[right_ind], depth + 1)

        return {
            'feature_index': best_feature,
            'split_value': best_value,
            'left': left_subtree,
            'right': right_subtree 
        }

    def Gini(self, left_y, right_y):
        n_left = len(left_y)
        n_right = len(right_y)
        total = n_left + n_right
        
        if total == 0:
            return 0
        
        p_left = np.sum(left_y == np.unique(left_y, return_counts=True)[1].max()) / n_left
        p_right = np.sum(right_y == np.unique(right_y, return_counts=True)[1].max()) / n_right

        return p_left * p_right

    def train(self, X, y):
        self.tree = self.build_tree(X, y, 0)

    def query(self, X):
        return [self.traverse(x, self.tree) for x in X]

    def traverse(self, x, node):
        if isinstance(node, dict):
            if x[node['feature_index']] < node['split_value']:
                return self.traverse(x, node['left'])
            else:
                return self.traverse(x, node['right'])
        else:
            return node


X_classification = np.array([
    [2.0, 3.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [3.0, 2.0],
    [4.0, 5.0],
    [5.0, 3.0],
    [2.0, 1.0],
    [3.0, 2.0],
    [1.0, 1.0],
    [6.0, 5.0]
])

y_classification = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])

# Example dataset for multi-class classification
X_multiclass = np.array([
    [2.0, 3.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [1.0, 2.0],
    [4.0, 5.0],
    [5.0, 3.0],
    [2.0, 1.0],
    [3.0, 2.0]
])

y_multiclass = np.array([0, 1, 2, 0, 2, 1, 0, 2])

# Example usage for binary classification
print("Binary Classification Example:")
dt_classifier_binary = DecisionTreeClassifier(max_depth=5)
dt_classifier_binary.train(X_classification, y_classification)

X_test_binary = np.array([
    [2.5, 3.5],
    [1.0, 1.0],
    [5.5, 4.5]
])

predictions_binary = dt_classifier_binary.query(X_test_binary)
print("Predictions:", predictions_binary)

# Example usage for multi-class classification
print("\nMulti-Class Classification Example:")
dt_classifier_multiclass = DecisionTreeClassifier(max_depth=5)
dt_classifier_multiclass.train(X_multiclass, y_multiclass)

X_test_multiclass = np.array([
    [2.5, 3.5],
    [1.0, 1.0],
    [4.5, 5.5]
])

predictions_multiclass = dt_classifier_multiclass.query(X_test_multiclass)
print("Predictions:", predictions_multiclass)