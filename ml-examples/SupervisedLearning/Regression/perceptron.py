import numpy as np

'''

NOTE:
We do not have to standardize the feature values in all cases, but it does avoid the issue
of exploding gradients along with adjusting the learning rate(lr).

'''

class Perceptron:
    def __init__(self):
        pass

    def get_data(self, data):
        X = np.array([np.array(x[:-1]) for x in data])
        y = np.array([x[-1] for x in data])
        return X, y

    def predict(self, X):
        X_std = self.get_stdX(X)
        approx = np.dot(X_std, self.weights) + self.bias
        return np.sign(approx)

    def get_stdX(self, X): return (X - self.mean) / self.std

    def set_stdX(self, X): self.mean, self.std = X.mean(axis=0), X.std(axis=0)

    def train(self, data, lr=1e-2, epochs=1000):
        X, y = self.get_data(data)
        self.set_stdX(X)
        X_std = self.get_stdX(X)
        n = X_std.shape[1]
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(epochs):
            for idx, x_i in enumerate(X_std):
                y_i = y[idx]
                cond = y_i * (np.dot(x_i, self.weights) + self.bias) <= 0
                if cond:
                    self.weights += lr * y_i * x_i
                    self.bias += lr * y_i
    
# Example usage
data = [
    (1,2,1),
    (2,3,1),
    (3,4,-1),
    (4,5,-1)
]

test_data = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5]
])

perceptron = Perceptron()
perceptron.train(data)
perceptron_predictions = perceptron.predict(test_data)
print("Perceptron predictions:", perceptron_predictions)