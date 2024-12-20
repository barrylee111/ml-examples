import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def best_split(self, X, y):
        best_mse, best_feature, best_value = float('inf'), None, None
        n = X.shape[1]

        for feature in range(n):
            feature_vals = np.unique(X[:, feature])
            for val in feature_vals:
                # inds
                left_inds = X[:, feature] < val
                right_inds = ~left_inds

                # labels / check
                left_y = y[left_inds]
                right_y = y[right_inds]

                if len(left_y) == 0 or len(right_y) == 0: continue
                
                # mse / update
                mse = self.MSE(y)
                if mse < best_mse:
                    best_mse, best_feature, best_value = mse, feature, val

        return best_feature, best_value

    def build_tree(self, X, y, depth):
        best_feature, best_value = self.best_split(X, y)
        if depth == self.max_depth or best_feature is None: return np.mean(y)
        
        # inds
        left_inds = X[:, best_feature] < best_value
        right_inds = ~left_inds

        # subtrees
        left_subtree = self.build_tree(X[left_inds], y[left_inds], depth+1)
        right_subtree = self.build_tree(X[right_inds], y[right_inds], depth+1)
        
        return {
            'feature_index': best_feature,
            'split_value': best_value,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def get_data(self, data):
        X = np.array([np.array(x[:-1]) for x in data])
        y = np.array([x[-1] for x in data])

        return X, y

    def MSE(self, y): return np.mean(np.square(y - np.mean(y)))

    def predict(self, X): return [self.traverse(x, self.tree) for x in X]

    def train(self, data):
        X, y = self.get_data(data)
        self.tree = self.build_tree(X, y, 0)

    def traverse(self, x, node):
        if isinstance(node, dict):
            if x[node['feature_index']] < node['split_value']: return self.traverse(x, node['left'])
            return self.traverse(x, node['right'])

        return node


# Usage examples
# Decision Tree Examples
tree_model = DecisionTree(max_depth=5)
data = np.array(
    [[1,2,2],
        [2,3,3],
        [4,5,5],
        [6,7,7],
        [7,8,8],
        [8,9,9]]
)

test_data = np.array(
    [[8, 9],
        [5, 6],
        [4, 5],
        [2, 3]]
)

tree_model.train(data)
predictions_regression = tree_model.predict(test_data)
print("Regression predictions:", predictions_regression)

data = np.array(
    [[1,2,1],
        [2,3,1],
        [3,4,1],
        [7,7,1],
        [7,8,0],
        [8,9,0]]
)

test_data = np.array(
    [[4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [2, 3]]
)

tree_model.train(data)

predictions_classification = tree_model.predict(test_data)
print("Classification predictions:", predictions_classification)