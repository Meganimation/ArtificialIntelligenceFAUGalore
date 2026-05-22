import numpy as np

X = np.array([[1, 0],
              [0, 1]])
W_Q = W_K = W_V = np.eye(2) 
d_k = 2

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    x_max[np.isinf(x_max)] = 0 
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

print("--- MASKED SELF-ATTENTION ---")

Q = X @ W_Q
K = X @ W_K
V = X @ W_V
S = (Q @ K.T) / np.sqrt(d_k)

print(f"1) Scaled Scores (S):\n{np.round(S, 4)}\n")

M = np.array([[0, -np.inf],
              [0, 1]])
masked_S = S + M

print(f"2) Masked Scaled Scores (S + M):\n{np.round(masked_S, 4)}\n")

A_self = softmax(masked_S)
Z = A_self @ V

print(f"3) Attention Weights:\n{np.round(A_self, 4)}")
print(f"   Z (Output of Masked Self-Attention):\n{np.round(Z, 4)}\n")
print("CROSS-ATTENTION!")

H = np.array([[2, 0],
              [0, 3]])
Q_cross = Z @ W_Q
K_cross = H @ W_K
V_cross = H @ W_V

print(f"4) Q_cross:\n{np.round(Q_cross, 4)}")
print(f"   K_cross & V_cross:\n{np.round(K_cross, 4)}\n")

S_cross = (Q_cross @ K_cross.T) / np.sqrt(d_k)
A_cross = softmax(S_cross)
Z_final = A_cross @ V_cross

print(f"5) Cross-Attention Scaled Scores:\n{np.round(S_cross, 4)}")
print(f"   Cross-Attention Weights:\n{np.round(A_cross, 4)}")
print(f"   Final Z (Output of Cross-Attention):\n{np.round(Z_final, 4)}")