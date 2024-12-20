import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=10, hidden_size=256, output_size=2):
        self.W1 = np.random.rand(input_size, hidden_size) * 1e-2
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, output_size) * 1e-2
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

    def forward(self, X_train):
        Z1 = np.dot(X_train, self.W1) + self.b1 # X * W1 + b1
        A1 = self.sigmoid(Z1)
        self.A1 = A1
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)

        return A2
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def sigmoid(self, z): return 1/(1+np.exp(-z))

    def sigmoid_derivative(self, z): return z * (1-z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def train(self, X_train, y_train, epochs: int=10000, lr=1e-2):
        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            loss = self.cross_entropy_loss(y_train, y_pred)
            self.backprop(X_train, y_train, y_pred, lr)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch} - Loss: {loss}')

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    iris = load_iris()
    X = iris.data
    y = iris.target

    # One-hot encode the target variable
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Initialize and train the neural network
    nn = NeuralNetwork(input_size=4, hidden_size=100, output_size=3)
    nn.train(X_train, y_train, 50000, 1.5e-3)

    # Make predictions
    predictions = nn.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)

    print("Predictions:", predictions)
    print("True Labels:", true_labels)
    accuracy = np.mean(predictions == true_labels)
    print("Accuracy:", accuracy)