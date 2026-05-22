import numpy as np

# Observed data
D = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
n = len(D)
k = np.sum(D)

# Assume a Beta(alpha, beta) prior. 
# We'll use alpha=2, beta=2 (adds 1 "virtual" success and 1 "virtual" failure)
alpha = 2
beta = 2

# Calculate the MAP estimate (mode of the posterior distribution)
map_mean = (k + alpha - 1) / (n + alpha + beta - 2)

print(f"Observed successes (k): {k}")
print(f"Total trials (n): {n}")
print(f"MAP Estimate with Beta(2,2) prior: {map_mean:.4f}")