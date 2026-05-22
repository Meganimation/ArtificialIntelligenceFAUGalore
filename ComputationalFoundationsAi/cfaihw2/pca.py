import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

#Todo:
# compute covariance matrix of ‘X_centered’ (refer to p. 20 in pca slides)
# compute eigen values and eigen vector (hint: use np.linalg.eigh)
# choose TWO largest ‘eigenvalues’ and their corresponding ‘eigenvectors’ 
# eigenvectors: two eigen vectors


# Project original data ‘X_centered’ onto two principal components (refer to p. 31 in pca slides)
# X_pca: ‘X_centered’ data projected onto two principal components

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
