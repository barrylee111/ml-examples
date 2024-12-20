import argparse
import numpy as np
from test import regression_tests

# KNN, LRN, LOG

class Model:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    ### UTILS ###

    def add_bias(self, X): return np.column_stack((np.ones(X.shape[0]), X))

    def get_data(self, data): return np.array([np.array(x[:-1]) for x in data]), np.array([x[-1] for x in data])

    def get_norm_data(self, X): return (X - self.mean) / self.std

    def set_norm_data(self, X): self.mean, self.std = X.mean(axis=0), X.std(axis=0)

    #############

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
                mse = self.MSE(left_y) + self.MSE(right_y)
                if mse < best_mse: best_mse, best_feature, best_value = mse, feature, val

        return best_feature, best_value

    def build_tree(self, X, y, depth):
        best_feature, best_value = self.best_split(X, y)

        # base cases
        if best_feature is None or depth == self.max_depth: return np.mean(y)

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

    def MSE(self, y):
        return np.mean(np.square(y - np.mean(y)))

    def predict(self, X): return [self.traverse(x, self.tree) for x in X]

    def train(self, data):
        X, y = self.get_data(data)
        self.tree = self.build_tree(X, y, 0)

    def traverse(self, x, node):
        if isinstance(node, dict):
            if x[node['feature_index']] < node['split_value']: return self.traverse(x, node['left'])
            return self.traverse(x, node['right'])

        return node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_name', help='Enter the name of model you intend to test')
    args = parser.parse_args()
    test_name = args.test_name
    regression_tests(test_name, Model)

if __name__ == '__main__':
    main()
