import numpy as np

# 0. Define Inputs and Weights
X = np.array([[1, 0, 1, 0],
              [0, 2, 0, 2]])

W_Q = np.array([[1, 0],
                [0, 1],
                [1, 0],
                [0, 1]])

W_K = np.array([[1, 1],
                [0, 1],
                [1, 1],
                [0, 1]])

W_V = np.array([[1, 0],
                [0, 2],
                [1, 0],
                [0, 2]])

# 1. Compute Q, K, and V
# Obtained by multiplying the input embeddings by the trained weight matrices
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("1) Matrices Q, K, V:")
print(f"Q:\n{Q}\n\nK:\n{K}\n\nV:\n{V}\n")

# 2. Compute Scaled Scores
# The score is calculated by taking the dot product of the query and key vectors
d_k = Q.shape[1] # Projection dimension is 2
scaled_scores = (Q @ K.T) / np.sqrt(d_k)

print("2) Scaled Attention Scores:")
print(np.round(scaled_scores, 4), "\n")

# 3. Apply Row-wise Softmax and compute Z
# Softmax normalizes the scores so they add up to 1
# Subtracting the row max is a standard stability practice to prevent overflow
exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=1, keepdims=True))
attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Final self-attention output is the weighted sum of the value vectors
Z = attention_weights @ V

print("3) Attention Weights (Softmax):")
print(np.round(attention_weights, 4), "\n")
print("Z (Output of Self-Attention):")
print(np.round(Z, 4))