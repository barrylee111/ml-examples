import argparse
import numpy as np
from test import dl_tests

# ANN, CNN (PyTorch), GAN (PyTorch)

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.rand(input_size, hidden_size) * 1e-2
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def backprop(self, X, y_true, y_pred, lr):
        m = y_true.shape[0]

        dZ2 = y_pred - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / m

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        self.A1 = A1
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)

        return A2

    def predict(self, X): return np.argmax(self.forward(X), axis=1)

    def sigmoid(self, z): return 1 / (1+np.exp(-z))

    def sigmoid_derivative(self, z): return z * (1-z)

    def softmax(self, z):
        exp_z = np.exp(z - np.sum(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train(self, X, y_true, epochs, lr):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.cross_entropy_loss(y_true, y_pred)
            self.backprop(X, y_true, y_pred, lr)

            if epoch % 1000 == 0: print(f'Epoch {epoch} - Loss: {loss}')
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_name', help='Enter the name of model you intend to test')
    args = parser.parse_args()
    test_name = args.test_name
    dl_tests(test_name, Model)

if __name__ == '__main__':
    main()