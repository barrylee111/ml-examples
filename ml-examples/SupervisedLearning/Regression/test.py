import numpy as np

def regression_tests(test, Model):
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

    test_data = np.array([
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
    ])

    if test == 'PER':
        # Example usage
        X = np.array([
            [1, 2], [2, 3], [3, 4], [4, 5]
        ])
        data = [
            (1,2,1),
            (2,3,1),
            (3,4,-1),
            (4,5,-1)
        ]

        perceptron = Model()
        perceptron.train(data)
        perceptron_predictions = perceptron.predict(X)
        print("Perceptron predictions:", perceptron_predictions)

    if test == 'SVM':
        # Example usage
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

    if test == 'SGD':
        # SGD Test
        lrm = Model()
        lrm.train(training_data)
        output = lrm.predict(test_data)
        print(f'SGD: {output}')
        print()


    if test == 'LRN':
        # Linear Regression Norm Testing
        lrm = Model()
        lrm.train(training_data)
        output = lrm.predict(test_data)
        print(f'LRN: {output}')
        print()


    if test == 'KNN':
        # KNN Examples
        data = np.array(
            [(2,3,2),
            (3,4,4),
            (5,6,6)])
        X_test = np.array([[7,8], [1,2], [4,5]])
        knn = Model()
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
        knn = Model()
        knn.train(data)
        output_in_range = knn.predict(X_test_in_range)
        output_out_of_range = knn.predict(X_test_out_of_range)

        print("KNN output for test data mostly in range:", output_in_range)
        print("KNN output for test data mostly out of range:", output_out_of_range)

    if test == 'TREE':
        # Decision Tree Examples
        tree_model = Model(max_depth=5)
        data = np.array(
            [[1,2,2],
            [2,3,3],
            [4,5,5],
            [6,7,7],
            [7,8,8],
            [8,9,9]]
        )

        test_data = np.array(
            [[8, 9],
            [5, 6],
            [4, 5],
            [2, 3]]
        )

        tree_model.train(data)
        predictions_regression = tree_model.predict(test_data)
        print("Regression predictions:", predictions_regression)

        data = np.array(
            [[1,2,1],
            [2,3,1],
            [3,4,1],
            [7,7,1],
            [7,8,0],
            [8,9,0]]
        )

        test_data = np.array(
            [[4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [2, 3]]
        )
        
        tree_model.train(data)
        
        predictions_classification = tree_model.predict(test_data)
        print("Classification predictions:", predictions_classification)

    if test == 'LOG':
        np.random.seed(0)

        def generate_data(n_samples=100, noise=0.1):
            # Generate random 2D points
            X = np.random.rand(n_samples, 2) * 2 - 1  # Scale to range [-1, 1]
            
            # Define a decision boundary (line) separating the two classes
            # For simplicity, let's use the line y = x
            decision_boundary = lambda x: x
            
            # Classify points based on their position relative to the decision boundary
            y = (X[:, 1] > decision_boundary(X[:, 0])).astype(int)
            
            # Add noise to the labels
            y_noisy = y ^ (np.random.rand(n_samples) < noise)
            
            return X, y_noisy

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

        lrm = Model()
        lrm.train(data)
        output = lrm.predict(test_data)
        print(output)

        # data = generate_data()
        # X_test, _ = generate_data(2)
        # lrm.train(data)
        # output = lrm.predict(X_test)
        # print()
        # print(output)