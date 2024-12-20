import numpy as np

class KNNClassifier():
    def __init__(self, k: int=5):
        self.k = k
    
    def query(self, X):
        predictions = np.array([])
        for x in X:
            distances = np.linalg.norm(x - self.train_x, axis=1)
            ind = np.argsort(distances)[:self.k]
            labels = self.train_y[ind]
            prediction = np.bincount(labels).argmax()
            predictions = np.append(predictions, prediction)

        return predictions

    def train(self, X, y): self.train_x, self.train_y = X, y

# Example usage:
X_train = np.array([[1, 2], [3, 4], [6, 5]])
y_train = np.array([1, 1, 0])

knn_classifier = KNNClassifier(k=2)
knn_classifier.train(X_train, y_train)

X_test = np.array([[2.0, 3.0], [5.0, 6.0]])
predictions = knn_classifier.query(X_test)

print("Predictions:", predictions)  # Output: [1.0, 0.0]
