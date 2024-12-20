import numpy as np

'''

NOTE:
We do not have to standardize the feature values in all cases, but it does avoid the issue
of exploding gradients along with adjusting the learning rate(lr).

'''

class LinearRegressionModel:
    def __init__(self):
        pass

    def add_bias(self, X): return np.column_stack((np.ones(X.shape[0]), X))

    def get_data(self, data):
        X = np.array([np.array(x[:-1]) for x in data])
        y = np.array([x[-1] for x in data])

        return X, y

    def predict(self, X):
        X = self.get_std_vals(X)
        X_b = self.add_bias(X)
        return X_b @ self.weights

    def get_std_vals(self, X): return (X - self.mean) / self.std
    
    def set_Xstd(self, X): self.mean, self.std = X.mean(axis=0), X.std(axis=0)
    
    def train(self, data):
        X, y = self.get_data(data)
        self.set_Xstd(X)
        X = self.get_std_vals(X)
        X_b = self.add_bias(X)
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y # (X.T*X)^-1*X.T*y
    
    def r_squared(self, X, y_true):
        y_pred = self.query(X)
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
# SGD Test
training_data = [
    (1200, 0, 1, 5, 240),
    (2000, 1, 2, 3, 500),
    (800, 0, 1, 7, 180),
    (2500, 2, 3, 1, 800),
    (1500, 0, 2, 4, 350),
    (2300, 1, 3, 6, 720),
    (1600, 0, 2, 2, 330),
    (1100, 0, 1, 4, 220),
    (1800, 1, 2, 5, 410),
    (2200, 2, 3, 7, 750),
    (1700, 0, 3, 1, 500),
    (1000, 0, 1, 3, 210),
    (1900, 1, 2, 2, 430),
    (2100, 0, 3, 6, 650),
    (1400, 1, 1, 5, 300)
]

# test data with price to predict
test_data = [
    (1500, 0, 2, 4),
    (2300, 1, 3, 6),
    (1200, 0, 1, 2),
    (1800, 1, 2, 3),
    (1100, 0, 1, 7),
    (2500, 2, 3, 5),
    (1600, 0, 2, 1),
    (2000, 1, 2, 4),
    (2100, 0, 3, 3),
    (1400, 1, 1, 6)
]

lrm = LinearRegressionModel()
lrm.train(training_data)
output = lrm.predict(test_data)
print(f'SGD: {output}')
print()