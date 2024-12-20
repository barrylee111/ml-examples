import numpy as np

def pca(X, num_components):
    # Standardize the data
    X_meaned = X - np.mean(X, axis=0)
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(X_meaned, rowvar=False)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Select a subset of the eigenvectors (num_components)
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    
    # Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    
    return X_reduced

# Example usage
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
pca(X, 1)
