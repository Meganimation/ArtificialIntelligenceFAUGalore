import numpy as np
import matplotlib.pyplot as plt

# Define the true function
# The true error function is sine + cosine
def true_function(X):
    return np.sin(X) + 2 * np.cos(X)

# =====================================================
# 8.1)-1: RBF (Radial Basis Function) Kernel
# =====================================================
# Computes the RBF (Gaussian) kernel between two vectors X1 and X2.
# X1 : numpy array 1D (size=n)
# X2 : numpy array 1D (size=m)
# returns:
# K : numpy array of shape (n, m)
# Formula: k(x_i, x_j) = exp(-||x_i - x_j||^2)

def rbf_kernel(X1, X2):
    '''
    8.1)-1: Implement RBF Kernel
    K(x,x) = [k(x_1,x_1)  ...  k(x_1,x_m)
               ...          ...    ...
              k(x_n,x_1)  ...  k(x_n,x_m)]
    where k(x_i, x_j) = exp(-||x_i - x_j||^2)
    '''
    # Compute pairwise squared Euclidean distances
    # X1: (n, 1), X2: (m, 1)
    # Expand dimensions for broadcasting
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    
    # Compute ||x_i - x_j||^2
    sq_distances = np.sum((X1 - X2) ** 2, axis=2) if X1.ndim > 2 else (X1 - X2) ** 2
    
    # Simpler approach: compute pairwise distances
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    
    # Create matrices for broadcasting
    X1_mat = X1_flat[:, np.newaxis]  # shape (n, 1)
    X2_mat = X2_flat[np.newaxis, :]  # shape (1, m)
    
    # Pairwise squared distances
    sq_dist = (X1_mat - X2_mat) ** 2
    
    # RBF kernel: exp(-distance^2)
    K = np.exp(-sq_dist)
    
    return K

# =====================================================
# 8.1)-2: Gaussian Process Regression
# =====================================================
def gaussian_regression(X_train, y_train, X_test):
    '''
    8.1)-2: Perform Gaussian Process Regression
    
    Given:
    - X_train: training inputs (n, 1)
    - y_train: training outputs (n,)
    - X_test: test inputs (m, 1)
    
    Compute:
    - mu*: mean of posterior predictive distribution
      mu* = k(x_test, x) * K(x,x)^(-1) * y
    - sigma*: covariance of posterior predictive distribution
      sigma* = k(x_test, x_test) - k(x_test, x) * K(x,x)^(-1) * k(x, x_test)
    '''
    # 1) Compute K(x, x) - covariance matrix of training data
    K_train = rbf_kernel(X_train, X_train)
    
    # 2) Compute K(x, x)^(-1) - inverse of training covariance
    K_train_inv = np.linalg.inv(K_train)
    
    # 3) Compute k(x_test, x) - covariance between test and training data
    k_test_train = rbf_kernel(X_test, X_train)  # shape (m, n)
    
    # 4) Compute k(x, x_test) - transpose of above
    k_train_test = k_test_train.T  # shape (n, m)
    
    # 5) Compute k(x_test, x_test) - covariance matrix of test data
    k_test_test = rbf_kernel(X_test, X_test)  # shape (m, m)
    
    # Compute posterior mean: mu* = k(x_test, x) * K(x,x)^(-1) * y
    mu_star = k_test_train @ K_train_inv @ y_train  # shape (m,)
    
    # Compute posterior covariance: sigma* = k(x_test, x_test) - k(x_test, x) * K(x,x)^(-1) * k(x, x_test)
    sigma_star = k_test_test - k_test_train @ K_train_inv @ k_train_test  # shape (m, m)
    
    return mu_star, sigma_star

# =====================================================
# Training Data
# =====================================================
print("=" * 70)
print("Bayesian Optimization using Gaussian Process Regression")
print("=" * 70)
print()

# 6 experiments with different X values
X_train = np.array([[1], [2], [4], [6], [8], [9]])
y_train = true_function(X_train).ravel()

print("8.1)-1: RBF Kernel Implementation")
print("-" * 70)
print("Training data X_train:", X_train.ravel())
print("Training data y_train:", y_train.round(4))
print()

# =====================================================
# 8.1)-3: Estimate mean and covariance for single test point
# =====================================================
print("8.1)-3: Gaussian Process Prediction for X_test = [2.2]")
print("-" * 70)

X_test_single = np.array([[2.2]])
mu_single, sigma_single = gaussian_regression(X_train, y_train, X_test_single)

# Compute standard deviation (sqrt of diagonal of covariance)
sd_single = np.sqrt(np.diag(sigma_single))

print(f"Test point: X_test = {X_test_single.ravel()[0]}")
print(f"Predicted mean μ*:  {mu_single[0]:.4f}")
print(f"Predicted std σ:    {sd_single[0]:.4f}")
print()

# =====================================================
# Question 2: Extended test set
# =====================================================
print("Question 2: Extended Test Set with 10 Points")
print("-" * 70)

X_test = np.linspace(0, 10, 10).reshape(-1, 1)
mean, sigma = gaussian_regression(X_train, y_train, X_test)

# Compute standard deviation for each test point
sd = np.sqrt(np.diag(sigma))

print("X_test values:", X_test.ravel())
print()
print("Predicted means μ*:")
print(mean.round(4))
print()
print("Predicted std devs σ:")
print(sd.round(4))
print()

# =====================================================
# Question 3: Lower Confidence Bound (LCB)
# =====================================================
print("Question 3: Lower Confidence Bound (LCB) Strategy")
print("-" * 70)

# Compute LCB for each test point
# LCB(x) = μ(x) - κ * σ(x), where κ is typically 1.96 for 95% confidence
kappa = 1.96
lcb = mean - kappa * sd

print(f"Using κ = {kappa} for 95% confidence interval")
print()
print("LCB values for each X_test:")
for i, (x, lcb_val, mu_val, s_val) in enumerate(zip(X_test.ravel(), lcb, mean, sd)):
    print(f"  X = {x:.2f}: LCB = {lcb_val:.4f} (μ = {mu_val:.4f}, σ = {s_val:.4f})")

# Find the next hyperparameter based on LCB
next_idx = np.argmin(lcb)
next_x = X_test[next_idx]

print()
print(f"→ Next hyperparameter to evaluate: X = {next_x[0]:.2f}")
print(f"  LCB value at this point: {lcb[next_idx]:.4f}")
print()

# =====================================================
# Question 4 (Extra Credit): Visualization
# =====================================================
print("Question 4: Gaussian Process Regression Visualization")
print("-" * 70)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_test, true_function(X_test), 'r:', linewidth=2, label="True function")
plt.plot(X_train, y_train, 'r.', markersize=12, label="Training data")
plt.plot(X_test, mean, 'b-', linewidth=2, label="GP Prediction (mean)")

# Confidence interval: ±1.96*σ (95% confidence)
plt.fill_between(X_test.ravel(), mean - 1.96*sd, mean + 1.96*sd, 
                 alpha=0.2, color='b', label="95% Confidence interval")

# Mark the next point to evaluate
plt.plot(next_x, true_function(next_x), 'g*', markersize=20, label="Next evaluation point")

plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Gaussian Process Regression for Bayesian Optimization', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✓ Visualization complete")
