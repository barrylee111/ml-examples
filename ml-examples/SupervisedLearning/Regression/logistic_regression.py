import numpy as np


'''

NOTE:
We do not have to standardize the feature values in all cases, but it does avoid the issue
of exploding gradients along with adjusting the learning rate(lr).

'''

class LogisticRegressionModel:
    def __init__(self):
        pass

    def add_bias(self, X): return np.column_stack((np.ones(X.shape[0]), X))

    def get_data(self, data):
        X = np.array([np.array(x[:-1]) for x in data])
        y = np.array([x[-1] for x in data])

        return X, y
    
    def predict(self, X):
        X_b = self.add_bias(X)
        return self.sigmoid(np.dot(X_b, self.weights))
    
    def sigmoid(self, z): return 1/(1+np.exp(-z))

    def train(self, data, lr=1e-2, epochs=1000):
        X, y = self.get_data(data)
        X_b = self.add_bias(X)
        m, n = X_b.shape
        self.weights = np.zeros(n)

        for _ in range(epochs):
            h = self.sigmoid(np.dot(X_b, self.weights))
            gradient = np.dot(X_b.T, (h-y)) / m # np.dot(X_b.T, (h-y)) / m
            self.weights -= lr * gradient

data = [
    (1,3,0),
    (2,4,0),
    (3,-1,1),
    (4,-2,1),
    (5,-3,1)
]

test_data = np.array([
    (7,-3),
    (2,5),
    (5,5)
])

lrm = LogisticRegressionModel()
lrm.train(data)
output = lrm.predict(test_data)
print(output)

'''
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000, regularization=None, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def initialize_weights(self, n_features):
        # Initialize weights for all features including bias
        self.weights = np.zeros(n_features)
    
    def add_bias(self, X):
        # Add bias term to X
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    
    def compute_cost(self, X, y):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, self.weights))
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        
        # Add regularization term
        if self.regularization == 'l2':
            reg_term = (self.lambda_reg / (2 * m)) * np.sum(self.weights[1:]**2)  # exclude bias term
            cost += reg_term
        elif self.regularization == 'l1':
            reg_term = (self.lambda_reg / m) * np.sum(np.abs(self.weights[1:]))  # exclude bias term
            cost += reg_term
        
        return cost
    
    def gradient_descent(self, X, y):
        m = X.shape[0]
        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, self.weights))
            error = h - y
            gradient = np.dot(X.T, error) / m
            
            # Regularization term in gradient descent update
            if self.regularization == 'l2':
                reg_term = (self.lambda_reg / m) * self.weights
                reg_term[0] = 0  # Do not regularize bias term
            elif self.regularization == 'l1':
                reg_term = (self.lambda_reg / m) * np.sign(self.weights)
                reg_term[0] = 0  # Do not regularize bias term
            else:
                reg_term = 0
            
            self.weights -= self.learning_rate * (gradient + reg_term)
    
    def fit(self, X, y):
        X = self.add_bias(X)
        n_features = X.shape[1]
        self.initialize_weights(n_features)
        self.gradient_descent(X, y)
    
    def predict(self, X):
        X = self.add_bias(X)
        predictions = self.sigmoid(np.dot(X, self.weights))
        return (predictions > 0.5).astype(int)

'''