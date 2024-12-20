import numpy as np

# 

class Model:
    def __init__(self):
        pass
        
test = 'Tree'

if test == 'KNN':
    X_train = np.array([[1, 2], [3, 4], [6, 5]])
    y_train = np.array([1, 1, 0])

    knn_classifier = Model(k=2)
    knn_classifier.train(X_train, y_train)

    X_test = np.array([[2.0, 3.0], [5.0, 6.0]])
    predictions = knn_classifier.query(X_test)

    print("Predictions:", predictions)  # Output: [1.0, 0.0]

if test == 'Tree':
    X_classification = np.array([
        [2.0, 3.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [3.0, 2.0],
        [4.0, 5.0],
        [5.0, 3.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [1.0, 1.0],
        [6.0, 5.0]
    ])

    y_classification = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])

    # Example dataset for multi-class classification
    X_multiclass = np.array([
        [2.0, 3.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [1.0, 2.0],
        [4.0, 5.0],
        [5.0, 3.0],
        [2.0, 1.0],
        [3.0, 2.0]
    ])

    y_multiclass = np.array([0, 1, 2, 0, 2, 1, 0, 2])

    # Example usage for binary classification
    print("Binary Classification Example:")
    dt_classifier_binary = Model(max_depth=5)
    dt_classifier_binary.train(X_classification, y_classification)

    X_test_binary = np.array([
        [2.5, 3.5],
        [1.0, 1.0],
        [5.5, 4.5]
    ])

    predictions_binary = dt_classifier_binary.query(X_test_binary)
    print("Predictions:", predictions_binary)

    # Example usage for multi-class classification
    print("\nMulti-Class Classification Example:")
    dt_classifier_multiclass = Model(max_depth=5)
    dt_classifier_multiclass.train(X_multiclass, y_multiclass)

    X_test_multiclass = np.array([
        [2.5, 3.5],
        [1.0, 1.0],
        [4.5, 5.5]
    ])

    predictions_multiclass = dt_classifier_multiclass.query(X_test_multiclass)
    print("Predictions:", predictions_multiclass)