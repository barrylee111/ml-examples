import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def dl_tests(test_name, Model):
    pass

    if test_name == 'NN':
        iris = load_iris()
        X = iris.data
        y = iris.target

        # One-hot encode the target variable
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y.reshape(-1, 1))

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

        # Initialize and train the neural network
        nn = Model(input_size=4, hidden_size=100, output_size=3)
        nn.train(X_train, y_train, 50000, 1.5e-3)

        # Make predictions
        predictions = nn.predict(X_test)
        true_labels = np.argmax(y_test, axis=1)

        print("Predictions:", predictions)
        print("True Labels:", true_labels)
        accuracy = np.mean(predictions == true_labels)
        print("Accuracy:", accuracy)