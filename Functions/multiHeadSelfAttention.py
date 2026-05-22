import numpy as np
Z_1 = np.array([[1.8885, 0.4465],
                [0.0069, 7.9723]])
Z_2 = np.array([[2.88, 0.24],
                [3.99, 0.01]])
W_O = np.array([[1, 1],
                [0, 1],
                [1, 1],
                [0, 1]])
Z_concat = np.concatenate((Z_1, Z_2), axis=1)

print("1) Concatenated Matrix (Z_concat):")
print(Z_concat, "\n")
Z_final = Z_concat @ W_O

print("2) Final Z Output (Z_concat * W_O):")
print(np.round(Z_final, 4))