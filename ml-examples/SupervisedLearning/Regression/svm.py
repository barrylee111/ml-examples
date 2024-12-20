import numpy as np


'''

NOTE:
We do not have to standardize the feature values in all cases, but it does avoid the issue
of exploding gradients along with adjusting the learning rate(lr).

'''

class Model:
    def __init__(self):
        pass

    def get_data(self, data):
        X = np.array([np.array(x[:-1]) for x in data])
        y = np.array([x[-1] for x in data])

        return X, y

    def predict(self, X):
        X = self.standardize(X)
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)
    
    def standardize(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def train(self, data, lr=1e-2, _lambda=1e-2, epochs=1000):
        X, y = self.get_data(data)
        X = self.standardize(X)
        n = X.shape[1]
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                y_i = y[idx]
                cond = y_i * (np.dot(x_i, self.weights) - self.bias) >= 1
                if cond:
                    self.weights -= lr * (2 * _lambda * self.weights)
                else:
                    self.weights -= lr * (2 * _lambda * self.weights - np.dot(x_i, y_i))
                    self.bias -= lr * y_i
    

data = [
    (1,2,1),
    (2,3,1),
    (3,4,-1),
    (4,5,-1)
]

test_data = [
    (0,1),
    (1,2),
    (2,3),
    (3,4),
    (7,8)
]

svm = Model()
svm.train(data)
svm_predictions = svm.predict(test_data)
print("SVM predictions:", svm_predictions)
