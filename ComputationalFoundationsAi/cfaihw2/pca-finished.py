import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Compute covariance matrix of 'X_centered'
cov_matrix = np.cov(X_centered.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Choose TWO largest eigenvalues and their corresponding eigenvectors
top_2_eigenvectors = eigenvectors[:, :2]

print(f"Top 2 Eigenvalues: {eigenvalues[:2]}")
print(f"Explained Variance: {eigenvalues[:2] / np.sum(eigenvalues) * 100}%")

# Project original data 'X_centered' onto two principal components
X_pca = X_centered @ top_2_eigenvectors

plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=iris.target_names[label], alpha=0.7)
plt.title('PCA on Iris Dataset (Top 2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
