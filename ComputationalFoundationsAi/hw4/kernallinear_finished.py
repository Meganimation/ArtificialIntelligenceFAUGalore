import numpy as np
import matplotlib.pyplot as plt

# 1D dataset
rng = np.random.default_rng(0)
X = np.linspace(-3, 3, 20)[:, None]
y = 0.5*np.sin(X).ravel() + 0.15 * rng.normal(size=X.shape[0])

def rbf_kernel(a, b, length_scale=1.0):
    sq_norm_a = np.sum(a**2, axis=1)[:, None]
    sq_norm_b = np.sum(b**2, axis=1)[None, :]
    sq_dist = sq_norm_a + sq_norm_b - 2 * (a @ b.T)
    return np.exp(-sq_dist / (2 * length_scale**2))
    # implementation of the RBF (Gaussian) kernel
    # a has shape (n, d) — n samples, each of d features,
    # b has shape (m, d) — m samples,
    # The function will return a kernel matrix (n,m).
    # ‘’’

def linear_kernel(a, b):
    # This is the inner product kernel corresponds to ordinary linear regression
    # The function will return a kernel matrix (n,m).
    # Returns inner product between a and b
    return a @ b.T

# Compute rbf kernel matrix ‘K’ using X
#  ‘’’
# K(X,X)=[■(k(x_0,x_0 )&…&k(x_0,x_n )@…&…&…@k(x_n,x_0 )&…&k(x_n,x_n ) )]
# Compute rbf kernel matrix ‘K’ using data X
# ‘’’
K = rbf_kernel(X, X)

lam = 1e-2

# compute alpha using K above
# formula of alpha is a little different from the slide.
# Ridge regression lam*np.eye(len(X)) is used here (Don’t worry about it)
alpha = np.linalg.solve(K + lam * np.eye(len(X)), y)

# Test data Xtest
Xtest = np.linspace(-3.5, 3.5, 200)[:, None]

# Predict
#  ‘’’
# # Compute K_small using rbf_kernel
# # Predict y_pred using K_small and alpha
# ‘’’
K_small = rbf_kernel(Xtest, X)
y_pred = K_small @ alpha

# Compute regular linear regression
K_lin = linear_kernel(X, X)
alpha_lin = np.linalg.solve(K_lin + lam * np.eye(len(X)), y)
y_pred_lin = linear_kernel(Xtest, X) @ alpha_lin

# Plot
plt.figure()
plt.scatter(X.ravel(), y, label="train")
plt.plot(Xtest.ravel(), y_pred, label="RBF kernel")
plt.plot(Xtest.ravel(), y_pred_lin, linestyle="--", label="Linear kernel")
plt.legend()
plt.title("Kernel Ridge Regression (tiny example)")
plt.show()
