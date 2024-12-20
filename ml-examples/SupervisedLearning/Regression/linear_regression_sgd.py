import numpy as np


class LinearRegressionSGDModel:
    def __init__(self):
        pass

    def add_bias(self, X): return np.column_stack((np.ones(X.shape[0]), X))

    def get_data(self, data):
        X = np.array([np.array(x[:-1], dtype=np.float64) for x in data])
        y = np.array([x[-1] for x in data])

        self.mean, self.std = X.mean(axis=0), X.std(axis=0)

        # norm data
        X = (X - self.mean) / self.std

        return X, y

    def query(self, data):
        X = np.array([np.array(d) for d in data])
        X = (X - self.mean) / self.std
        X_b = self.add_bias(X)
        return X_b @ self.weights
    
    def train(self, data, lr=1.5e-3, threshold=1e-6, epochs=10000):
        X, y = self.get_data(data)
        X_b = self.add_bias(X)
        print(X_b)
        # self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        m, n = X_b.shape
        self.weights = np.zeros(n)

        for _ in range(epochs):
            total_error = 0
            for idx in range(m):
                pred = np.dot(X_b[idx], self.weights)
                error = pred - y[idx]
                total_error += abs(error)
                gradient = X_b[idx] * error
                self.weights -= lr * gradient

            cost = total_error / m
            if cost < threshold: break

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

lrm = LinearRegressionSGDModel()
lrm.train(training_data)
output = lrm.query(test_data)
print(f'SGD: {output}')
print()