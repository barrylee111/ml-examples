import numpy as np


class KNNModel():
    def __init__(self, k: int=3):
        self.k = k

    # for reference
    # def euc_dist(self, x1, x2):
    #     return np.sqrt(np.sum((x1 - x2)**2))

    def get_data(self, data):
        X = np.array([np.array(x[:-1]) for x in data])
        y = np.array([x[-1] for x in data])

        return X, y

    def predict(self, X):
        predictions = np.array([])
        for x in X:
            distances = np.linalg.norm(x - self.train_x, axis=1)
            inds_n = np.argsort(distances)[:self.k]
            labels_n = self.train_y[inds_n]
            prediction = np.mean(labels_n)
            predictions = np.append(predictions, prediction)
        
        return predictions

    def train(self, data):
        X, y = self.get_data(data)
        self.train_x, self.train_y = X, y
        
# KNN Examples
knn = KNNModel()

data = np.array(
    [(2,3,2),
    (3,4,4),
    (5,6,6)])
X_test = np.array([[7,8], [1,2], [4,5]])

knn.train(data)
output = knn.predict(X_test)
print(f'KNN output: {output}')

data = np.array(
    [
        (10,20,20),
        (15,25,25),
        (20,30,30),
        (25,35,35),
        (30,40,40),
        (35,45,45),
        (40,50,50),
        (45,55,55),
        (50,60,60),
        (55,65,65)
    ]
)

X_test_in_range = np.array([[12, 22], [17, 27], [22, 32], [27, 37], [32, 42], [37, 47], [42, 52]])  # Mostly in range
X_test_out_of_range = np.array([[5, 15], [60, 70], [70, 80]])  # Out of range

knn.train(data)
output_in_range = knn.predict(X_test_in_range)
output_out_of_range = knn.predict(X_test_out_of_range)

print("KNN output for test data mostly in range:", output_in_range)
print("KNN output for test data mostly out of range:", output_out_of_range)